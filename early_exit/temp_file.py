
########################################
########################################
########################################
##### PROTBERT ######

########################################
########################################
## Classification

########################################!~!~!~!
# Normal
@R.register("tasks.Classification_walltime_ProtBert")
class Classification_walltime_ProtBert(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('auprc@micro', 'f1_max'), verbose=0, num_class=1, weight=None, tokenizer=AutoTokenizer):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer.
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(Classification_walltime_ProtBert, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.metric = metric
        self.tokenizer=tokenizer

    # ------------- helper to space‑separate sequences ------------
    @staticmethod
    def _prep_protbert(seqs):               # EDIT: name
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned

    # --------------------------- PREDICT -------------------------
    def predict(self, batch, all_loss=None, metric=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()        # ensure correct GPU
        self.mlp.to(device).eval()

        graphs = batch["graph"]

        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]
        batch_size = len(sequences)

        # ---- tokenise ----
        prepared = self._prep_protbert(sequences)
        enc = self.tokenizer(
            prepared,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,                # ProtBERT was trained up to 1024
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"     ].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # ---- temperatures / bookkeeping ----
        n_layers = self.model.config.num_hidden_layers
        layer_stop = int(os.getenv("LAYER"))

        # ---- build extended attention mask like BERT does ----
        ext_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # ---- initial embeddings ----
        hs = self.model.embeddings(input_ids)

        # ---- iterate through encoder layers (no weight sharing) ----
        for layer_idx, layer in enumerate(self.model.encoder.layer):     # EDIT: simple loop
            # BERT layer forward
            out = layer(
                hs,
                attention_mask=ext_mask,
                head_mask=None,
                output_attentions=False
            )
            hs = out[0]                 # first element is hidden states

            if layer_idx == layer_stop:
            # ---------- classifier ----------
                pooled   = hs.mean(dim=1)
                final_logits   = self.mlp[layer_idx](pooled)
                break

        return {
            "pred": final_logits,
        }

    # ------------- unchanged target / evaluate -----------------
    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        result = {}
        pred  = preds["pred"]
        target = target.to(pred.device)

        metric_out = {}
        for m in self.metric:
            if m == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif m == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif m == "f1_max":
                score = metrics.f1_max(pred, target)
                metric_out["f1_max"] = score
            else:
                raise ValueError(f"Unknown metric {m}")
        return metric_out

########################################!~!~!~!
# Exit

@R.register("tasks.EarlyExitClassification_walltime_ProtBert")
class EarlyExitClassification_walltime_ProtBert(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('auprc@micro', 'f1_max'), verbose=0, num_class=1, weight=None, tokenizer=AutoTokenizer):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer.
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(EarlyExitClassification_walltime_ProtBert, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.metric = metric
        self.tokenizer=tokenizer

    # ------------- helper to space‑separate sequences ------------
    @staticmethod
    def _prep_protbert(seqs):               # EDIT: name
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned

    # --------------------------- PREDICT -------------------------
    def predict(self, batch, all_loss=None, metric=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()        # ensure correct GPU
        self.mlp.to(device).eval()

        graphs = batch["graph"]

        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]
        batch_size = len(sequences)

        # ---- tokenise ----
        prepared = self._prep_protbert(sequences)
        enc = self.tokenizer(
            prepared,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,                # ProtBERT was trained up to 1024
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"     ].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # ---- temperatures / bookkeeping ----
        n_layers = self.model.config.num_hidden_layers
        temps    = torch.ones(n_layers, device=device)
        threshold = float(os.getenv("THRESHOLD", "0.0"))

        final_logits = [None] * batch_size
        final_layers = [None] * batch_size
        best_prob    = torch.full((batch_size,), -float("inf"), device=device)
        best_logits  = [None] * batch_size
        best_layers  = [None] * batch_size
        computed_layers = [None] * batch_size  
        active       = torch.arange(batch_size, device=device)

        # ---- build extended attention mask like BERT does ----
        ext_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # ---- initial embeddings ----
        hs = self.model.embeddings(input_ids)

        # ---- iterate through encoder layers (no weight sharing) ----
        for layer_idx, layer in enumerate(self.model.encoder.layer):     # EDIT: simple loop
            if len(active) == 0:
                break
            for idx in active.tolist():
                computed_layers[idx] = layer_idx

            # slice to active samples only
            hs_active = hs[active]

            # BERT layer forward
            out = layer(
                hs_active,
                attention_mask=ext_mask[active],
                head_mask=None,
                output_attentions=False
            )
            hs_active = out[0]                 # first element is hidden states
            hs[active] = hs_active

            # ---------- classifier ----------
            pooled   = hs_active.mean(dim=1)
            logits   = self.mlp[layer_idx](pooled)
            prob     = torch.sigmoid(logits / temps[layer_idx])
            max_p, _ = prob.max(dim=1)

            # ---------- best‑so‑far ----------
            is_final = layer_idx == n_layers - 1
            better   = max_p > best_prob[active]
            if is_final and os.getenv("SELECT_LAST", "False") == "True":
                better = torch.ones_like(better, dtype=torch.bool)

            if better.any():
                g_idx = active[better]
                best_prob[g_idx] = max_p[better]
                for j, gi in enumerate(g_idx.tolist()):
                    best_logits[gi] = logits[better][j]
                    best_layers[gi] = layer_idx

            # ---------- early‑exit ----------
            exit_mask  = max_p > threshold
            newly_exit = active[exit_mask]
            still_act  = active[~exit_mask]

            for j, gi in enumerate(newly_exit.tolist()):
                final_logits[gi] = logits[exit_mask][j]
                final_layers[gi] = layer_idx

            active = still_act

        # ---- force exit any stragglers ----
        for gi in active.tolist():
            final_logits[gi] = best_logits[gi]
            final_layers[gi] = best_layers[gi]

        preds = torch.stack(final_logits, dim=0)

        # legacy 2000‑wide ascii tensor (unchanged)
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in sequences],
            batch_first=True, padding_value=0,
        )
        if ascii_mat.size(1) < 2000:
            pad = ascii_mat.new_zeros(ascii_mat.size(0), 2000 - ascii_mat.size(1))
            ascii_mat = torch.cat([ascii_mat, pad], dim=1)

        return {
            "pred": preds,
            "layers": torch.tensor(final_layers, device=device, dtype=torch.int64),
            "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64),
            "sequences": ascii_mat,
        }

    # ------------- unchanged target / evaluate -----------------
    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        result = {}
        pred  = preds["pred"]
        layers = preds["layers"]
        computed_layers = preds["computed_layers"]
        sequences = preds["sequences"]
        target = target.to(pred.device)

        metric_out = {}
        for m in self.metric:
            if m == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif m == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif m == "f1_max":
                score = metrics.f1_max(pred, target)
                metric_out["f1"] = score
            else:
                raise ValueError(f"Unknown metric {m}")
            metric_out[tasks._get_metric_name(m)] = score

        freq = torch.bincount(layers.cpu())
        avg_layer = (torch.arange(len(freq), device=freq.device) * freq).sum() / freq.sum()

        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed


        with open(os.getenv("RESULT_PICKLE"), "wb") as f:
            pickle.dump(
                {"preds": pred, "target": target, "layers": layers, "avg_computed_layer": average_computed_layer,
                 "metric": metric_out, "sequences": sequences}, f
            )

        result["avg_layer"] = avg_layer.item()
        result["avg_computed_layer"] = average_computed_layer.item()
        result.update(metric_out)
        return result
    

########################################!~!~!~!
# Analysis

########################################
########################################
## Property

########################################!~!~!~!
# Normal

@R.register("tasks.Property_continuous_protbert")
class Property_continuous_protbert(tasks.Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super(Property_continuous_protbert, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_layers = model.num_layers
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        for param in self.model.parameters():
            param.requires_grad = False


    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    @staticmethod
    def _prep_protbert(seqs):
        """Space‑separate, upper‑case, map U/O→X (ProtBERT convention)."""
        out = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            out.append(" ".join(list(s)))
        return out
        
    def predict(self, batch, all_loss=None, metric=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()        # ensure correct GPU
        self.mlp.to(device).eval()
        #print(f"device {device}")
        graphs  = batch["graph"]

        # ---- graphs → raw sequences -------------------------------------
        seqs = ["".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
                for g in graphs]
        B = len(seqs)

        # ---- temperatures / threshold ------------------------------------
        layer_out = float(os.getenv("LAYER"))
        # ---- tokenizer ----------------------------------------------------
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_bert", do_lower_case=False, use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache")
            )
        enc = self._tokenizer(
            self._prep_protbert(seqs),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]

        # ---- initial embeddings ------------------------------------------
        hs = self.model.embeddings(input_ids)                 # (B,L,H)
        ext_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0

        # ---- iterate through 30 encoder layers (no sharing) --------------
        for lidx, layer in enumerate(self.model.encoder.layer):
            # forward only for still‑active samples
            hs = layer(
                hs,
                attention_mask=ext_mask,
                head_mask=None,
                output_attentions=False
            )[0]
            if lidx == layer_out:
                pooled = hs.mean(dim=1)
                final_log = self.mlp[lidx](pooled)
                break
            # (B,C)

        return {
            "pred": final_log,
        }

 
    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, preds, target):
        pred = preds["pred"]
        target = target.to(pred.device)
        print(f"{pred.shape} pred.shape")
        print(f"self.num class {self.num_class}")
        labeled = ~torch.isnan(target)
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
                acc = score.item()
                metric["acc"] = acc
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)
        return metric



########################################!~!~!~!
# Exit

@R.register("tasks.EarlyExitProperty_continuous_protbert")
class EarlyExitProperty_continuous_protbert(tasks.Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super(EarlyExitProperty_continuous_protbert, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_layers = model.num_layers
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    @staticmethod
    def _prep_protbert(seqs):
        """Space‑separate, upper‑case, map U/O→X (ProtBERT convention)."""
        out = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            out.append(" ".join(list(s)))
        return out

    def predict(self, batch, all_loss=None, metric=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()        # ensure correct GPU
        self.mlp.to(device).eval()
        #print(f"device {device}")
        graphs  = batch["graph"]

        # ---- graphs → raw sequences -------------------------------------
        seqs = ["".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
                for g in graphs]
        B = len(seqs)

        # ---- temperatures / threshold ------------------------------------
        thr = float(os.getenv("THRESHOLD", "0.0"))
        tmp_file = os.getenv("TEMPERATURE_FILE")
        if tmp_file and tmp_file.lower() != "none":
            temps = torch.tensor(self._extract_temperatures(tmp_file), device=device)
        else:
            temps = torch.ones(self.num_layers, device=device)

        # ---- tokenizer ----------------------------------------------------
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_bert", do_lower_case=False, use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache")
            )
        enc = self._tokenizer(
            self._prep_protbert(seqs),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]

        # ---- initial embeddings ------------------------------------------
        hs = self.model.embeddings(input_ids)                 # (B,L,H)
        ext_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0

        # ---- bookkeeping -------------------------------------------------
        final_log = [None] * B
        final_lay = [None] * B
        best_prob = torch.full((B,), -float("inf"), device=device)
        best_log  = [None] * B
        best_lay  = [None] * B
        computed_layers = [None] * B 
        active    = torch.arange(B, device=device)
         # ---- iterate through 30 encoder layers (no sharing) --------------
        for lidx, layer in enumerate(self.model.encoder.layer):
            if active.numel() == 0:
                break
            for idx in active.tolist():
                computed_layers[idx] = lidx 
            # forward only for still‑active samples
            hs_act = layer(
                hs[active],
                attention_mask=ext_mask[active],
                head_mask=None,
                output_attentions=False
            )[0]
            hs[active] = hs_act

            # task head
            pooled = hs_act.mean(dim=1)
            logits = self.mlp[lidx](pooled)
            prob   = torch.sigmoid(logits / temps[lidx])
            max_p  = prob.max(dim=1).values

            # best‑so‑far update
            is_final = lidx == self.num_layers - 1
            better   = max_p > best_prob[active]
            if is_final and os.getenv("SELECT_LAST", "False") == "True":
                better = torch.ones_like(better, dtype=torch.bool)
            if better.any():
                gi = active[better]
                best_prob[gi] = max_p[better]
                for k, gidx in enumerate(gi.tolist()):
                    best_log[gidx] = logits[better][k]
                    best_lay[gidx] = lidx

            # early‑exit decision
            exit_mask = max_p > thr
            newly_exit, still = active[exit_mask], active[~exit_mask]

            for k, gidx in enumerate(newly_exit.tolist()):
                final_log[gidx] = logits[exit_mask][k]
                final_lay[gidx] = lidx

            active = still

        # ---- force‑exit leftovers ----------------------------------------
        for gidx in active.tolist():
            final_log[gidx] = best_log[gidx]
            final_lay[gidx] = best_lay[gidx]

        preds = torch.stack(final_log, dim=0)                 # (B,C)

        # legacy 2000‑wide ASCII tensor
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in seqs],
            batch_first=True, padding_value=0
        )
        if ascii_mat.size(1) < 2000:
            ascii_mat = torch.cat(
                [ascii_mat, ascii_mat.new_zeros(B, 2000 - ascii_mat.size(1))],
                dim=1
            )

        return {
            "pred": preds,
            "layers": torch.tensor(final_lay, device=device, dtype=torch.int64),
            "sequences": ascii_mat, 
            "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64)
        }

 
    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, preds, target):
        pred = preds["pred"]
        layers = preds["layers"]
        sequences = preds["sequences"]
        print(f"{pred.shape} pred.shape")
        print(f"self.num class {self.num_class}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target = target.to(device)
        labeled = ~torch.isnan(target)
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
                acc = score.item()
                metric["acc"] = acc
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

        
        layer_frequencies = torch.bincount(layers)
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total
        metric["layer"] = average_layer

        computed_layers = preds["computed_layers"]
        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed
        metric["avg_computed_layer"] = average_computed_layer.item()

        results_pickle = os.getenv("RESULT_PICKLE")
        results = {"preds": pred, "target": target, "layers": layers, "avg_computed_layer": average_computed_layer.item(), "metric": metric, "sequences": sequences}

        with open(results_pickle, 'wb') as f:
            pickle.dump(results, f)

        return metric


########################################!~!~!~!
# Analysis


@R.register("tasks.EarlyExitProperty_continuous_protbert_analysis")
class EarlyExitProperty_continuous_protbert_analysis(
    EarlyExitProperty_continuous_protbert
):
    """Analysis‑only variant that records *each* layer’s max‑probability and
    predicted label for ProtBERT. Output structure matches
    `EarlyExitProperty_walltime_analysis`."""

    # ─────────────────────── predict ──────────────────────────────────────────
    def predict(self, batch, all_loss=None, metric=None):
        print("starting prediction")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                              # <- use engine‑assigned GPU
        self.model.to(device).eval()
        self.mlp.to(device).eval()

        # ----- raw sequences -------------------------------------------------
        graphs = batch["graph"]
        seqs = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]
        B = len(seqs)

        # ----- temperatures --------------------------------------------------
        tmp_file = os.getenv("TEMPERATURE_FILE")
        if tmp_file and tmp_file.lower() != "none":
            temps = torch.tensor(self.extract_temperatures(tmp_file), device=device)
        else:
            temps = torch.ones(self.num_layers, device=device)

        # ----- tokenizer ------------------------------------------------------
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_bert", do_lower_case=False, use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache"),
            )
        enc = self._tokenizer(
            self._prep_protbert(seqs),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]

        # ----- embeddings ----------------------------------------------------
        hs = self.model.embeddings(input_ids)            # (B,L,H)
        ext_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0

        # ----- storage tensors ----------------------------------------------
        layer_pred_prob  = torch.full((self.num_layers, B), float("nan"), device=device)
        layer_pred_label = torch.full((self.num_layers, B), -1, dtype=torch.long, device=device)

        # ----- iterate -------------------------------------------------------
        for idx, layer in enumerate(self.model.encoder.layer):
            hs = layer(
                hs,
                attention_mask=ext_mask,
                head_mask=None,
                output_attentions=False,
            )[0]

            pooled = hs.mean(dim=1)
            logits = self.mlp[idx](pooled)
            probs  = torch.sigmoid(logits / temps[idx])

            max_prob, pred_lbl = probs.max(dim=1)
            layer_pred_prob[idx]  = max_prob
            layer_pred_label[idx] = pred_lbl

        # ----- ASCII matrix (legacy) ----------------------------------------
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in seqs],
            batch_first=True, padding_value=0,
        )
        if ascii_mat.size(1) < 2000:
            ascii_mat = torch.cat(
                [ascii_mat, ascii_mat.new_zeros(B, 2000 - ascii_mat.size(1))], dim=1
            )

        return {
            "sequences": ascii_mat,
            "layer_pred_prob": layer_pred_prob.t().contiguous(),   # (B,L)
            "layer_pred_label": layer_pred_label.t().contiguous(), # (B,L)
        }

    # ─────────────────────── evaluate ───────────────────────────────────────
    def evaluate(self, preds, target):
        print("starting evaluation")
        layer_pred_prob  = preds["layer_pred_prob"]
        layer_pred_label = preds["layer_pred_label"]
        true_labels = target[:, 0].long().to(layer_pred_label.device)
        layer_correct = layer_pred_label.eq(true_labels.unsqueeze(1))

        # dumps ----------------------------------------------------------------
        res_pickle = os.getenv("RESULT_PICKLE")
        if res_pickle:
            with open(res_pickle, "wb") as f:
                pickle.dump({
                    "layer_pred_prob": layer_pred_prob.cpu(),
                    "layer_pred_label": layer_pred_label.cpu(),
                    "layer_correct": layer_correct.cpu(),
                }, f)
        csv_path = os.getenv("RESULT_CSV")
        if csv_path:
            with open(csv_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["sample_idx", "layer_idx", "probability", "correct"])
                B, L = layer_pred_prob.shape
                for s in range(B):
                    for l in range(L):
                        writer.writerow([
                            s, l,
                            layer_pred_prob[s, l].item(),
                            int(layer_correct[s, l].item()),
                        ])
        return {}




########################################!~!~!~!
########################################
## Node


########################################!~!~!~!
# Normal
@R.register("tasks.ClassificationTemperature_Node_continuous_ProtBert")
class ClassificationTemperature_Node_continuous_ProtBert(tasks.Task, core.Configurable):
    _option_members = {"criterion", "metric"}

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("macro_auprc", "macro_auroc"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        verbose=0,
    ):
        super(ClassificationTemperature_Node_continuous_ProtBert, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, disable normalization
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose
        self.num_layers = 33

    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    @staticmethod
    def _prep_protbert(seqs):
        """Space‑separate, upper‑case, map U/O→X (ProtBERT convention)."""
        out = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            out.append(" ".join(list(s)))
        return out

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute mean, std, and num_class on the training set,
        then build an MLP head per layer.
        """
        # Determine whether we are working at the node, atom, or residue level
        self.view = getattr(train_set[0]["graph"], "view", "atom")

        # Collect all target values from the train set for statistics
        values_list = []
        for data in train_set:
            values_list.append(data["graph"].target)  # shape: (num_nodes,) or (num_residues,)

        values = torch.cat(values_list, dim=0)
        mean = values.float().mean()
        std = values.float().std()

        # Figure out number of classes if doing classification
        num_class = 1
        if values.dtype == torch.long:
            # If max label is >1 or not using BCE, it means multiclass
            nmax = values.max().item()
            if nmax > 1 or "bce" not in self.criterion:
                nmax += 1
            num_class = nmax

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

    def predict(self, batch, all_loss=None, metric=None):
        """
        Forward the batch through ProtBERT up to `last_layer`
        (taken from the env-var LAYER).  Return per-residue logits
        from that layer only.

        Returns
        -------
        dict
            {"pred": final_logits}
        """
        # -----------------------------------------------------------
        # 0) set-up
        # -----------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()
        self.mlp.to(device).eval()

        # -----------------------------------------------------------
        # 1) graphs → raw sequences
        # -----------------------------------------------------------
        graphs = batch["graph"]
        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]

        # -----------------------------------------------------------
        # 2) tokenise
        # -----------------------------------------------------------
        if not hasattr(self, "_tokenizer"):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_bert",
                do_lower_case=False,
                use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache"),
            )

        enc = self._tokenizer(
            self._prep_protbert(sequences),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]

        # -----------------------------------------------------------
        # 3) embeddings + bert-style mask
        # -----------------------------------------------------------
        hs = self.model.embeddings(input_ids)                 # (B,L,H)
        ext_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0

        # -----------------------------------------------------------
        # 4) figure out which layer to stop at
        # -----------------------------------------------------------
        n_layers = self.model.config.num_hidden_layers
        env_val   = os.getenv("LAYER", str(n_layers - 1))     # default = last layer
        last_layer = int(float(env_val))                      # allow "3" or "3.0"

        if not (0 <= last_layer < n_layers):
            raise ValueError(f"LAYER={last_layer} out of range (0-{n_layers-1})")

        # -----------------------------------------------------------
        # 5) forward only up to that layer
        # -----------------------------------------------------------
        with torch.no_grad():
            for lidx, layer in enumerate(self.model.encoder.layer):
                hs = layer(
                    hs,
                    attention_mask=ext_mask,
                    head_mask=None,
                    output_attentions=False,
                )[0]
                if lidx == last_layer:
                    break

        # -----------------------------------------------------------
        # 6) per-sample logits via the *same* layer’s MLP head
        # -----------------------------------------------------------
        final_logits = []
        for b, seq in enumerate(sequences):
            seq_len = len(seq)
            h_i   = hs[b, 1 : seq_len + 1, :]      # drop [CLS]
            log_i = self.mlp[last_layer](h_i)      # (L_res,C)
            final_logits.append(log_i)

        # -----------------------------------------------------------
        # 7) return only what you asked for
        # -----------------------------------------------------------
        return {"pred": final_logits}

    def target(self, batch):
        """
        Return a dictionary with:
        "label": the node-level target
        "mask": a boolean mask indicating which nodes are labeled
        "size": used for some metrics requiring per-graph aggregates
        """
        graph = batch["graph"]
        size = graph.num_nodes if self.view in ["node", "atom"] else graph.num_residues
        return {
            "label": graph.target,   # shape: (num_nodes,) or (num_residues,)
            "mask": graph.mask,      # shape: (num_nodes,) or (num_residues,)
            "size": size
        }

    def evaluate(self, preds, target):
        """
        Evaluate each layer's predictions given the `target`.
        preds: list of tensors, each is shape (N, num_class) or (N,) depending on your MLP output
        target: dict with { "label", "mask", "size" }
        """
        metric = {}
        _target = target["label"]
        _mask = target["mask"]
        labeled = ~torch.isnan(_target) & _mask
        _size = functional.variadic_sum(labeled.long(), target["size"])
        pred = preds["pred"]

        device = pred.device if hasattr(pred, 'device') else (
        pred["pred"].device if isinstance(pred, dict) and "pred" in pred else 
        (pred[0].device if isinstance(pred, list) and len(pred) > 0 else torch.device("cpu"))
        )

        # Move target to the same device as pred
        _target = _target.to(device)
        labeled = labeled.to(device)
        _size = _size.to(device)

        for _metric in self.metric:
            if _metric in ["mae", "rmse"]:
                # Typically for regression
                if _metric == "mae":
                    score = F.l1_loss(pred, _target, reduction="none")
                else:  # rmse
                    score = F.mse_loss(pred, _target, reduction="none").sqrt()

                score = functional.masked_mean(score, labeled, dim=0)

            elif _metric in ["micro_auroc", "micro_auprc"]:
                # Single "micro" approach across all labeled nodes
                if _metric == "micro_auroc":
                    score = metrics.area_under_roc(pred[labeled], _target[labeled])
                else:
                    score = metrics.area_under_prc(pred[labeled], _target[labeled])

            elif _metric in ["macro_auroc", "macro_auprc"]:
                # "macro" means compute per-graph, then average
                if _metric == "macro_auroc":
                    score = metrics.variadic_area_under_roc(pred[labeled], _target[labeled], _size).mean()
                else:
                    score = metrics.variadic_area_under_prc(pred[labeled], _target[labeled], _size).mean()

            elif _metric == "macro_acc":
                # One typical approach for multi-class:
                # (pred[labeled].argmax(-1) == _target[labeled]).float()
                #print(f"labeled.shape {labeled.shape}")
                #print(f"target shape {_target.shape}")
                #pred = torch.cat(pred, dim=0)
                pred = torch.cat(pred, dim=0)
                #print(f"pred.shape {pred.shape}")
                pred_argmax = pred[labeled].argmax(dim=-1)
                correct = (pred_argmax == _target[labeled]).float()
                score = functional.variadic_mean(correct, _size).mean()
                metric["macro_acc"] = score.item()

            else:
                raise ValueError(f"Unknown metric `{_metric}`")

        return metric


########################################!~!~!~!
# Exit
@R.register("tasks.EarlyExitClassificationTemperature_Node_continuous_ProtBert")
class EarlyExitClassificationTemperature_Node_continuous_ProtBert(tasks.Task, core.Configurable):
    _option_members = {"criterion", "metric"}

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("macro_auprc", "macro_auroc"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        verbose=0,
    ):
        super(EarlyExitClassificationTemperature_Node_continuous_ProtBert, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, disable normalization
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose
        self.num_layers = 33

    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    @staticmethod
    def _prep_protbert(seqs):
        """Space‑separate, upper‑case, map U/O→X (ProtBERT convention)."""
        out = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            out.append(" ".join(list(s)))
        return out

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute mean, std, and num_class on the training set,
        then build an MLP head per layer.
        """
        # Determine whether we are working at the node, atom, or residue level
        self.view = getattr(train_set[0]["graph"], "view", "atom")

        # Collect all target values from the train set for statistics
        values_list = []
        for data in train_set:
            values_list.append(data["graph"].target)  # shape: (num_nodes,) or (num_residues,)

        values = torch.cat(values_list, dim=0)
        mean = values.float().mean()
        std = values.float().std()

        # Figure out number of classes if doing classification
        num_class = 1
        if values.dtype == torch.long:
            # If max label is >1 or not using BCE, it means multiclass
            nmax = values.max().item()
            if nmax > 1 or "bce" not in self.criterion:
                nmax += 1
            num_class = nmax

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

    def predict(self, batch, all_loss=None, metric=None):
        """
        Same inputs/outputs as the original ESM‑2 version, but revised to use
        ProtBERT.  The overall control‑flow, early‑exit logic, and return
        structure are unchanged so downstream code keeps working.
        """
        # ---------------------------------------------------------------
        # 0) bookkeeping & set‑up
        # ---------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()
        self.mlp.to(device).eval()

        graphs = batch["graph"]
        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]
        B = len(sequences)

        # ---------------------------------------------------------------
        # 1) temperatures / thresholds
        # ---------------------------------------------------------------
        threshold = float(os.getenv("CFG_THRESHOLD"))
        percent   = float(os.getenv("PERCENT"))

        tmp_file = os.getenv("TEMPERATURE_FILE")
        if tmp_file and tmp_file.lower() != "none":
            temps = torch.tensor(self.extract_temperatures(tmp_file), device=device)
        else:
            # number of encoder layers (30 for ProtBERT‑B)
            n_layers = getattr(self, "num_layers", self.model.config.num_hidden_layers)
            temps = torch.ones(n_layers, device=device)

        # ---------------------------------------------------------------
        # 2) tokenize with ProtBERT tokenizer (slow‑tokenizer, no lower‑case)
        # ---------------------------------------------------------------
        if not hasattr(self, "_tokenizer"):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_bert", do_lower_case=False, use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache"),
            )

        enc = self._tokenizer(
            self._prep_protbert(sequences),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]  # (B,L)

        # ---------------------------------------------------------------
        # 3) initial embeddings (B,L,H) and Bert‑style attention mask
        # ---------------------------------------------------------------
        hs = self.model.embeddings(input_ids)                               # (B,L,H)
        ext_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0          # (B,1,1,L)

        # ---------------------------------------------------------------
        # 4) placeholders for results & trackers (match original output)
        # ---------------------------------------------------------------
        final_logits = [None] * B
        final_layers = torch.full((B,), -1, device=device)
        best_logits  = [None] * B
        best_prob    = torch.full((B,), -float("inf"), device=device)
        best_layers  = torch.full((B,), -1, device=device)
        computed_layers = torch.full((B,), -1, device=device) 
        active       = torch.arange(B, device=device)

        n_layers = getattr(self, "num_layers", self.model.config.num_hidden_layers)

        # ---------------------------------------------------------------
        # 5) iterate over ProtBERT encoder layers
        # ---------------------------------------------------------------
        for lidx, layer in enumerate(self.model.encoder.layer):
            if active.numel() == 0:
                break
            for idx in active.tolist():
                computed_layers[idx] = lidx 

            # ---- forward pass ONLY for active samples -----------------
            hs_act = layer(
                hs[active],                   # (A,L,H)
                attention_mask=ext_mask[active],
                head_mask=None,
                output_attentions=False,
            )[0]
            hs[active] = hs_act               # write‑back

            # ---- MLP head per sample ----------------------------------
            logits_list = []
            max_prob_vec = torch.empty(active.size(0), device=device)

            for loc, gidx in enumerate(active.tolist()):
                seq_len = len(sequences[gidx])
                # slice off the [CLS] token at pos 0; keep only residues
                h_i = hs_act[loc, 1 : seq_len + 1, :]      # (L_res,H)
                log_i = self.mlp[lidx](h_i)                # (L_res,C)
                logits_list.append(log_i)

                probs = torch.sigmoid(log_i / temps[lidx])
                max_prob_vec[loc] = probs.max(dim=1).values.mean()

                if (probs.max(dim=1).values > threshold).float().mean() >= percent:
                    final_logits[gidx] = log_i
                    final_layers[gidx] = lidx

            # ---- split finished vs. still active ----------------------
            done_mask   = final_layers[active] != -1
            newly_done  = active[done_mask]
            still_active = active[~done_mask]
            is_final    = lidx == n_layers - 1

            if still_active.numel() > 0:
                better = max_prob_vec[~done_mask] > best_prob[still_active]
                if is_final and os.getenv("SELECT_LAST", "False") == "True":
                    better = torch.ones_like(better, dtype=torch.bool)

                if better.any():
                    upd_idx = still_active[better]
                    best_prob[upd_idx]   = max_prob_vec[~done_mask][better]
                    best_layers[upd_idx] = lidx
                    # map local → global for logits list
                    for k, g in enumerate(upd_idx.tolist()):
                        best_logits[g] = logits_list[(~done_mask).nonzero(as_tuple=True)[0][k]]

            active = still_active

        # ---------------------------------------------------------------
        # 6) force‑exit any leftovers -----------------------------------
        for g in active.tolist():
            final_logits[g] = best_logits[g]
            final_layers[g] = best_layers[g]

        # ---------------------------------------------------------------
        # 7) legacy ASCII‑matrix (2000‑wide) ----------------------------
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in sequences],
            batch_first=True,
            padding_value=0,
        )
        if ascii_mat.size(1) < 2000:
            pad = ascii_mat.new_zeros(ascii_mat.size(0), 2000 - ascii_mat.size(1))
            ascii_mat = torch.cat([ascii_mat, pad], dim=1)

        return {
            "pred": final_logits,               # list(Tensor[L_i,C])
            "layers": final_layers,             # Tensor[B]
            "sequences": ascii_mat,            # Tensor[B,2000]
            "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64)
        }

    def target(self, batch):
        """
        Return a dictionary with:
          "label": the node-level target
          "mask": a boolean mask indicating which nodes are labeled
          "size": used for some metrics requiring per-graph aggregates
        """
        graph = batch["graph"]
        size = graph.num_nodes if self.view in ["node", "atom"] else graph.num_residues
        return {
            "label": graph.target,   # shape: (num_nodes,) or (num_residues,)
            "mask": graph.mask,      # shape: (num_nodes,) or (num_residues,)
            "size": size
        }

    def evaluate(self, preds, target):
        """
        Evaluate each layer's predictions given the `target`.
        preds: list of tensors, each is shape (N, num_class) or (N,) depending on your MLP output
        target: dict with { "label", "mask", "size" }
        """
        metric = {}
        _target = target["label"]
        _mask = target["mask"]
        labeled = ~torch.isnan(_target) & _mask
        _size = functional.variadic_sum(labeled.long(), target["size"])
        pred = preds["pred"]

        layers = preds["layers"]
        layer_frequencies = torch.bincount(layers)
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total

        for _metric in self.metric:
            if _metric in ["mae", "rmse"]:
                # Typically for regression
                if _metric == "mae":
                    score = F.l1_loss(pred, _target, reduction="none")
                else:  # rmse
                    score = F.mse_loss(pred, _target, reduction="none").sqrt()

                score = functional.masked_mean(score, labeled, dim=0)

            elif _metric in ["micro_auroc", "micro_auprc"]:
                # Single "micro" approach across all labeled nodes
                if _metric == "micro_auroc":
                    score = metrics.area_under_roc(pred[labeled], _target[labeled])
                else:
                    score = metrics.area_under_prc(pred[labeled], _target[labeled])

            elif _metric in ["macro_auroc", "macro_auprc"]:
                # "macro" means compute per-graph, then average
                if _metric == "macro_auroc":
                    score = metrics.variadic_area_under_roc(pred[labeled], _target[labeled], _size).mean()
                else:
                    score = metrics.variadic_area_under_prc(pred[labeled], _target[labeled], _size).mean()

            elif _metric == "macro_acc":
                # One typical approach for multi-class:
                # (pred[labeled].argmax(-1) == _target[labeled]).float()
                #print(f"labeled.shape {labeled.shape}")
                #print(f"target shape {_target.shape}")
                #pred = torch.cat(pred, dim=0)
                pred = torch.cat(pred, dim=0)
                #print(f"pred.shape {pred.shape}")
                pred_argmax = pred[labeled].argmax(dim=-1)
                correct = (pred_argmax == _target[labeled]).float()
                score = functional.variadic_mean(correct, _size).mean()
                metric["macro_acc"] = score.item()

            else:
                raise ValueError(f"Unknown metric `{_metric}`")
            metric["layer"] = average_layer.item()

        computed_layers = preds["computed_layers"]
        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed
        metric["avg_computed_layer"] = average_computed_layer.item()

        return metric


########################################!~!~!~!
# Analysis


########################################
########################################
########################################
##### PROTALBERT #####


########################################
########################################
## Classification

########################################!~!~!~!
# Normal

@R.register("tasks.Classification_walltime_ProtAlbert") #--> old, was used for first max
class Classification_walltime_ProtAlbert(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('auprc@micro', 'f1_max'), verbose=0, num_class=1, weight=None, tokenizer=AutoTokenizer):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer.
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(Classification_walltime_ProtAlbert, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.metric = metric
        self.tokenizer=tokenizer

    @staticmethod
    def _prep_protalbert(seqs):
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned

    def predict(self, batch, all_loss=None, metric=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"device {device}")
        graphs = batch["graph"]
        self.model.to(device).eval() 
        self.mlp.to(device).eval()


        # Convert graphs -> sequences
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            seq = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(seq)
                              

        # 1) Tokenize once
        prepared = self._prep_protalbert(sequences)                       # EDIT
        enc = self.tokenizer(
            prepared,
            add_special_tokens=True,
            padding=True,
            truncation=True,          # ProtAlbert max = 512
            return_tensors="pt",
            max_length = 550,
        )
        input_ids      = enc["input_ids"     ].to(device)
        attention_mask = enc["attention_mask"].to(device)


        # 5) Temperatures
        layer_out = int(os.getenv("LAYER"))
        # temperature_file = os.getenv("TEMPERATURE_FILE")
        # if temperature_file is not None and temperature_file != 'None':
        #     temperatures = self.extract_temperatures(temperature_file)
        #     temperatures = torch.tensor(temperatures, device=device)
        # else:
        n_layers  = self.model.config.num_hidden_layers

        hs = self.model.embeddings(input_ids)
        hs = self.model.encoder.embedding_hidden_mapping_in(hs)
        attn_ext = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        groups = self.model.encoder.albert_layer_groups
        layers_per_group = n_layers // self.model.config.num_hidden_groups
        global_layer_idx = 0
        layer_int = 0
        for group in groups:
            for _ in range(layers_per_group):

                # ---- forward one logical layer ----
                                               # (N,L,H)
                hs = group(
                    hs,
                    attention_mask=attn_ext,
                    head_mask=[None] * self.model.config.num_hidden_layers,  # ← FIX HERE
                    output_attentions=False,
                )
                hs = hs[0] if isinstance(hs, tuple) else hs

                # ---- classifier ----
                if layer_int == layer_out:
                    pooled = hs.mean(dim=1)                                        # (N,H)
                    logits = self.mlp[global_layer_idx](pooled)
                layer_int += 1
                global_layer_idx += 1
        return {
            "pred": logits,
        }

    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        result = {}
        pred = preds["pred"]
        target = target.to(pred.device)
        metric = {}
        f1_max = None
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
                f1_max = score
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)
            name = tasks._get_metric_name(_metric)
            metric[name] = score
        result["f1_max"] = f1_max.item()
        return result
    


########################################!~!~!~!
# Exit
@R.register("tasks.EarlyExitClassification_walltime_ProtAlbert") #--> old, was used for first max
class EarlyExitClassification_walltime_ProtAlbert(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('auprc@micro', 'f1_max'), verbose=0, num_class=1, weight=None, tokenizer=AutoTokenizer):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer.
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(EarlyExitClassification_walltime_ProtAlbert, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.metric = metric
        self.tokenizer=tokenizer

    @staticmethod
    def _prep_protalbert(seqs):
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned

    def predict(self, batch, all_loss=None, metric=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graphs = batch["graph"]
        self.model.to(device).eval() 
        self.mlp.to(device).eval()


        # Convert graphs -> sequences
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            seq = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(seq)

        batch_size = len(sequences)

        # We'll store final chosen logits + chosen layer
        final_logits = [None] * batch_size
        final_layers = [None] * batch_size
        best_prob     = torch.full((batch_size,), -float("inf"), device=device)   # NEW
        best_logits   = [None] * batch_size                                       # NEW
        best_layers   = [None] * batch_size      
        computed_layers = [None] * batch_size
        active = torch.arange(batch_size, device=device)                                 

        # 1) Tokenize once
        prepared = self._prep_protalbert(sequences)                       # EDIT
        enc = self.tokenizer(
            prepared,
            add_special_tokens=True,
            padding=True,
            truncation=True,          # ProtAlbert max = 512
            return_tensors="pt",
            max_length = 550,
        )
        input_ids      = enc["input_ids"     ].to(device)
        attention_mask = enc["attention_mask"].to(device)


        # 5) Temperatures
        threshold = float(os.getenv("THRESHOLD"))
        # temperature_file = os.getenv("TEMPERATURE_FILE")
        # if temperature_file is not None and temperature_file != 'None':
        #     temperatures = self.extract_temperatures(temperature_file)
        #     temperatures = torch.tensor(temperatures, device=device)
        # else:
        n_layers  = self.model.config.num_hidden_layers
        temps = torch.ones(n_layers, device=device)
        

        hs = self.model.embeddings(input_ids)
        hs = self.model.encoder.embedding_hidden_mapping_in(hs)
        attn_ext = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        groups = self.model.encoder.albert_layer_groups
        layers_per_group = n_layers // self.model.config.num_hidden_groups
        global_layer_idx = 0

        for group in groups:
            for _ in range(layers_per_group):
                if len(active) == 0:
                    break
                for idx in active.tolist():
                    computed_layers[idx] = global_layer_idx

                # ---- forward one logical layer ----
                hs_active = hs[active]                                                # (N,L,H)
                hs_active = group(
                    hs_active,
                    attention_mask=attn_ext[active],
                    head_mask=[None] * self.model.config.num_hidden_layers,  # ← FIX HERE
                    output_attentions=False,
                )
                hs_active = hs_active[0] if isinstance(hs_active, tuple) else hs_active
                hs[active] = hs_active

                # ---- classifier ----
                pooled = hs_active.mean(dim=1)                                        # (N,H)
                logits = self.mlp[global_layer_idx](pooled)
                prob   = torch.sigmoid(logits / temps[global_layer_idx])
                max_p, _ = prob.max(dim=1)

                # ---- best‑so‑far bookkeeping ----
                is_final = global_layer_idx == n_layers - 1
                better   = max_p > best_prob[active]
                if is_final and os.getenv("SELECT_LAST", "False") == "True":
                    better = torch.ones_like(better, dtype=torch.bool)

                if better.any():
                    g_idx = active[better]
                    best_prob[g_idx] = max_p[better]
                    for j, gi in enumerate(g_idx.tolist()):
                        best_logits[gi] = logits[better][j]
                        best_layers[gi] = global_layer_idx

                # ---- early‑exit decision ----
                exit_mask   = max_p > threshold
                newly_exit  = active[exit_mask]
                still_act   = active[~exit_mask]

                for j, gi in enumerate(newly_exit.tolist()):
                    final_logits[gi] = logits[exit_mask][j]
                    final_layers[gi] = global_layer_idx

                active = still_act
                global_layer_idx += 1
            if len(active) == 0:
                break

        # ---------- FORCE EXIT REMAINDERS ----------
        for gi in active.tolist():
            final_logits[gi] = best_logits[gi]
            final_layers[gi] = best_layers[gi]

        # ---------- STACK & RETURN ----------
        preds = torch.stack(final_logits, dim=0)
        # keep legacy 2000‑wide ASCII tensor (unchanged)
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in sequences],
            batch_first=True, padding_value=0,
        )
        if ascii_mat.size(1) < 2000:
            pad = ascii_mat.new_zeros(ascii_mat.size(0), 2000 - ascii_mat.size(1))
            ascii_mat = torch.cat([ascii_mat, pad], dim=1)

        return {
            "pred": preds,
            "layers": torch.tensor(final_layers, device=device, dtype=torch.int64),
            "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64),
            "sequences": ascii_mat,
        }

    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        result = {}
        pred = preds["pred"]
        layers = preds["layers"]
        sequences = preds["sequences"]
        target = target.to(pred.device)
        metric = {}
        layer_frequencies = torch.bincount(layers)
        f1_max = None
        for layer_idx, freq in enumerate(layer_frequencies):
            print(f"Layer {layer_idx}: Frequency {freq.item()}")
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
                f1_max = score
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)
            name = tasks._get_metric_name(_metric)
            metric[name] = score
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total

        computed_layers = preds["computed_layers"]
        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed

        # results_file = os.getenv("RESULT_FILE")
        # with open(results_file, 'a') as f:
        #    writer = csv.writer(f)
        #    writer.writerow([os.getenv("THRESHOLD"), f1_max.item(), average_layer.item()])
        results_pickle = os.getenv("RESULT_PICKLE")
        results = {"preds": pred, "target": target, "layers": layers, "avg_computed_layer": average_computed_layer, "metric": metric, "sequences": sequences}
        with open(results_pickle, 'wb') as f:
            pickle.dump(results, f)
        result["f1"] = f1_max.item()
        result["avg_layer"] = average_layer.item()
        result["avg_computed_layer"] = average_computed_layer.item()
        return result
    


########################################!~!~!~!
# Analysis

########################################
########################################
## Property

########################################!~!~!~!
# Normal


@R.register("tasks.Property_continuous_protalbert")
class Property_continuous_protalbert(tasks.Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super(Property_continuous_protalbert, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_layers = model.num_layers
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        for param in self.model.parameters():
            param.requires_grad = False


    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    @staticmethod
    def prep_protalbert(seqs):
        """
        ProtAlbert expects space‑separated, upper‑case single‑letter codes,
        with unknown/selenocysteine (‘U’, ‘O’) mapped to ‘X’.
        """
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned
        
    def predict(self, batch, all_loss=None, metric=None):
        """
        Forward pass with early‑exit for ProtAlbert.

        Returns:
            dict(pred = Tensor[B,C],
                layers = Tensor[B],
                sequences = Tensor[B,2000]  # legacy ASCII‑matrix)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()        # ensure correct GPU
        self.mlp.to(device).eval()
        graphs   = batch["graph"]
        seqs = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            seqs.append(sequence)

        #self.mlp.to(self.device)
        # ── Temperatures & confidence threshold ────────────────────────────────────
        layer_out = float(os.getenv("LAYER"))


        # ──Tokenizer────────────────────────────────────────────────────────────────
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_albert",  do_lower_case=False, use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache"))
        enc = self._tokenizer(
            self.prep_protalbert(seqs),
            add_special_tokens=True,
            padding=True,
            truncation=True,                # ProtAlbert can handle up to 512 aa
            max_length=550,
            return_tensors="pt",
        ).to(device)

        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]

        # ──Initial embeddings───────────────────────────────────────────────────────
        hs = self.model.embeddings(input_ids)                     # (B,L,E_word)
        hs = self.model.encoder.embedding_hidden_mapping_in(hs)   # (B,L,H)
        attn_ext = (1.0 - attn_mask[:, None, None, :]) * -10000.0 # (B,1,1,L)

        # ──Book‑keeping for early exit──────────────────────────────────────────────
        n_layers   = self.model.config.num_hidden_layers

        # ──Iterate through Albert layer‑groups──────────────────────────────────────
        groups          = self.model.encoder.albert_layer_groups
        layers_per_grp  = n_layers // self.model.config.num_hidden_groups
        g_layer_idx     = 0                                                # 0‑based

        layer_idx = 0
        for grp in groups:
            for _ in range(layers_per_grp):

                # ---- forward one logical layer ----
                hs = grp(
                    hs,
                    attention_mask=attn_ext,
                    head_mask=[None]*n_layers,
                    output_attentions=False,
                )[0]                              # tuple -> Tensor (N,L,H)


                # ---- task head ----
                if layer_idx == layer_out:
                    pooled  = hs.mean(dim=1)                   # (N,H)
                    logits  = self.mlp[g_layer_idx](pooled)
                    break
                layer_idx += 1
                g_layer_idx += 1

        return {
            "pred": logits
        }
 
    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, preds, target):
        pred = preds["pred"]
        device   = self.device
        pred = pred.to(device)
        target = target.to(device)
        labeled = ~torch.isnan(target)
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
                acc = score.item()
                metric["acc"] = acc
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)
        return metric

########################################!~!~!~!
# Exit

@R.register("tasks.EarlyExitProperty_continuous_protalbert")
class EarlyExitProperty_continuous_protalbert(tasks.Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super(EarlyExitProperty_continuous_protalbert, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_layers = model.num_layers
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        for param in self.model.parameters():
            param.requires_grad = False


    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    @staticmethod
    def prep_protalbert(seqs):
        """
        ProtAlbert expects space‑separated, upper‑case single‑letter codes,
        with unknown/selenocysteine (‘U’, ‘O’) mapped to ‘X’.
        """
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned
        
    def predict(self, batch, all_loss=None, metric=None):
        """
        Forward pass with early‑exit for ProtAlbert.

        Returns:
            dict(pred = Tensor[B,C],
                layers = Tensor[B],
                sequences = Tensor[B,2000]  # legacy ASCII‑matrix)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()        # ensure correct GPU
        self.mlp.to(device).eval()
        graphs   = batch["graph"]
        seqs = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            seqs.append(sequence)
        B        = len(seqs)
        #self.mlp.to(self.device)
        # ── Temperatures & confidence threshold ────────────────────────────────────
        thr = float(os.getenv("THRESHOLD"))
        tmp_file = os.getenv("TEMPERATURE_FILE")
        if tmp_file and tmp_file.lower() != "none":
            temps = torch.tensor(self.extract_temperatures(tmp_file), device=device)
        else:
            temps = torch.ones(self.model.config.num_hidden_layers, device=device)

        # ──Tokenizer────────────────────────────────────────────────────────────────
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_albert",  do_lower_case=False, use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache"))
        enc = self._tokenizer(
            self.prep_protalbert(seqs),
            add_special_tokens=True,
            padding=True,
            truncation=True,                # ProtAlbert can handle up to 512 aa
            max_length=550,
            return_tensors="pt",
        ).to(device)

        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]

        # ──Initial embeddings───────────────────────────────────────────────────────
        hs = self.model.embeddings(input_ids)                     # (B,L,E_word)
        hs = self.model.encoder.embedding_hidden_mapping_in(hs)   # (B,L,H)
        attn_ext = (1.0 - attn_mask[:, None, None, :]) * -10000.0 # (B,1,1,L)

        # ──Book‑keeping for early exit──────────────────────────────────────────────
        n_layers   = self.model.config.num_hidden_layers
        final_log  = [None] * B
        final_lay  = [None] * B
        best_prob  = torch.full((B,), -float("inf"), device=device)
        best_log   = [None] * B
        best_lay   = [None] * B
        computed_layers = [None] * B 
        active     = torch.arange(B, device=device)

        # ──Iterate through Albert layer‑groups──────────────────────────────────────
        groups          = self.model.encoder.albert_layer_groups
        layers_per_grp  = n_layers // self.model.config.num_hidden_groups
        g_layer_idx     = 0                                                # 0‑based

        for grp in groups:
            for _ in range(layers_per_grp):
                if active.numel() == 0:
                    break
                for idx in active.tolist():
                    computed_layers[idx] = g_layer_idx

                # ---- forward one logical layer ----
                hs_act = grp(
                    hs[active],
                    attention_mask=attn_ext[active],
                    head_mask=[None]*n_layers,
                    output_attentions=False,
                )[0]                              # tuple -> Tensor (N,L,H)
                hs[active] = hs_act

                # ---- task head ----
                pooled  = hs_act.mean(dim=1)                   # (N,H)
                logits  = self.mlp[g_layer_idx](pooled)
                prob    = torch.sigmoid(logits / temps[g_layer_idx])
                max_p   = prob.max(dim=1).values

                # ---- track best so far ----
                is_final = g_layer_idx == n_layers - 1
                better   = max_p > best_prob[active]
                if is_final and os.getenv("SELECT_LAST", "False") == "True":
                    better = torch.ones_like(better, dtype=torch.bool)

                if better.any():
                    gi = active[better]
                    best_prob[gi] = max_p[better]
                    for k, gidx in enumerate(gi.tolist()):
                        best_log[gidx] = logits[better][k]
                        best_lay[gidx] = g_layer_idx

                # ---- early‑exit decision ----
                exit_mask = max_p > thr
                newly_exit, still = active[exit_mask], active[~exit_mask]

                for k, gidx in enumerate(newly_exit.tolist()):
                    final_log[gidx] = logits[exit_mask][k]
                    final_lay[gidx] = g_layer_idx

                active = still
                g_layer_idx += 1
            if active.numel() == 0:
                break

        # ──force exit any remainder─────────────────────────────────────────────────
        for gidx in active.tolist():
            final_log[gidx] = best_log[gidx]
            final_lay[gidx] = best_lay[gidx]

        preds = torch.stack(final_log, dim=0)                       # (B,C)

        # legacy 2000‑wide ASCII encoding (unchanged)
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in seqs],
            batch_first=True, padding_value=0,
        )
        if ascii_mat.size(1) < 2000:
            pad = ascii_mat.new_zeros(ascii_mat.size(0), 2000 - ascii_mat.size(1))
            ascii_mat = torch.cat([ascii_mat, pad], dim=1)

        return {
            "pred": preds,
            "layers": torch.tensor(final_lay, device=device, dtype=torch.int64),
            "sequences": ascii_mat,
            "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64)
        }
 
    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, preds, target):
        pred = preds["pred"]
        layers = preds["layers"]
        sequences = preds["sequences"]
        print(f"{pred.shape} pred.shape")
        print(f"self.num class {self.num_class}")
        device   = self.device
        pred = pred.to(device)
        target = target.to(device)
        labeled = ~torch.isnan(target)
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
                acc = score.item()
                metric["acc"] = acc
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)
            

        layer_frequencies = torch.bincount(layers)
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total
        metric["layer"] = average_layer.item()

        computed_layers = preds["computed_layers"]
        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed
        metric["avg_computed_layer"] = average_computed_layer.item()

        results_pickle = os.getenv("RESULT_PICKLE")
        results = {"preds": pred, "target": target, "layers": layers, "avg_computed_layer": average_computed_layer.item(), "metric": metric, "sequences": sequences}

        with open(results_pickle, 'wb') as f:
            pickle.dump(results, f)

        return metric


########################################!~!~!~!
# Analysis
@R.register("tasks.EarlyExitProperty_continuous_protalbert_analysis")
class EarlyExitProperty_continuous_protalbert_analysis(EarlyExitProperty_continuous_protalbert):
    """Analysis‑only variant that records *every* layer's prediction for ProtAlbert."""

    def predict(self, batch, all_loss=None, metric=None):
        print("predicting")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        self.model.to(device).eval()
        self.mlp.to(device).eval()

        graphs = batch["graph"]
        seqs = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]
        B = len(seqs)
        n_layers = self.model.config.num_hidden_layers

        tmp_file = os.getenv("TEMPERATURE_FILE")
        if tmp_file and tmp_file.lower() != "none":
            temps = torch.tensor(self.extract_temperatures(tmp_file), device=device)
        else:
            temps = torch.ones(n_layers, device=device)

        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_albert", do_lower_case=False, use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache")
            )
        enc = self._tokenizer(
            self.prep_protalbert(seqs),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=550,
            return_tensors="pt",
        ).to(device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]

        hs = self.model.embeddings(input_ids)           # (B,L,E_word)
        hs = self.model.encoder.embedding_hidden_mapping_in(hs)  # (B,L,H)
        attn_ext = (1.0 - attn_mask[:, None, None, :]) * -10000.0

        # storage tensors [L,B]
        layer_pred_prob  = torch.full((n_layers, B), float("nan"), device=device)
        layer_pred_label = torch.full((n_layers, B), -1, dtype=torch.long, device=device)

        groups = self.model.encoder.albert_layer_groups
        layers_per_grp = n_layers // self.model.config.num_hidden_groups
        g_layer_idx = 0

        for grp in groups:
            for _ in range(layers_per_grp):
                hs = grp(
                    hs,
                    attention_mask=attn_ext,
                    head_mask=[None]*n_layers,
                    output_attentions=False,
                )[0]                                   # (B,L,H)

                pooled = hs.mean(dim=1)
                logits = self.mlp[g_layer_idx](pooled)
                probs  = torch.sigmoid(logits / temps[g_layer_idx])

                max_prob, pred_lbl = probs.max(dim=1)
                layer_pred_prob[g_layer_idx]  = max_prob
                layer_pred_label[g_layer_idx] = pred_lbl

                g_layer_idx += 1
                if g_layer_idx >= n_layers:
                    break

        # legacy ASCII matrix
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in seqs],
            batch_first=True, padding_value=0,
        )
        if ascii_mat.size(1) < 2000:
            pad = ascii_mat.new_zeros(ascii_mat.size(0), 2000 - ascii_mat.size(1))
            ascii_mat = torch.cat([ascii_mat, pad], dim=1)

        return {
            "sequences"      : ascii_mat,
            "layer_pred_prob": layer_pred_prob.t().contiguous(),   # (B,L)
            "layer_pred_label": layer_pred_label.t().contiguous(), # (B,L)
        }

    def evaluate(self, preds, target):
        print("evaluating")
        layer_pred_prob  = preds["layer_pred_prob"]  # (B,L)
        layer_pred_label = preds["layer_pred_label"] # (B,L)
        true_labels = target[:, 0].long().to(layer_pred_label.device)
        layer_correct = layer_pred_label.eq(true_labels.unsqueeze(1))

        # optional pickle/csv dump
        res_pickle = os.getenv("RESULT_PICKLE")
        if res_pickle:
            with open(res_pickle, "wb") as f:
                pickle.dump({
                    "layer_pred_prob" : layer_pred_prob.cpu(),
                    "layer_pred_label": layer_pred_label.cpu(),
                    "layer_correct"   : layer_correct.cpu(),
                }, f)

        csv_path = os.getenv("RESULT_CSV")
        if csv_path:
            with open(csv_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["sample_idx", "layer_idx", "probability", "correct"])
                B, L = layer_pred_prob.shape
                for s in range(B):
                    for l in range(L):
                        writer.writerow([
                            s,
                            l,
                            layer_pred_prob[s, l].item(),
                            int(layer_correct[s, l].item()),
                        ])
        return {}



########################################
########################################
## Node

########################################!~!~!~!
# Normal

@R.register("tasks.ClassificationTemperature_Node_continuous_ProtAlbert")
class ClassificationTemperature_Node_continuous_ProtAlbert(tasks.Task, core.Configurable):
    _option_members = {"criterion", "metric"}

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("macro_auprc", "macro_auroc"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        verbose=0,
    ):
        super(ClassificationTemperature_Node_continuous_ProtAlbert, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, disable normalization
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose


    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    @staticmethod
    def prep_protalbert(seqs):
        """
        ProtAlbert expects space‑separated, upper‑case single‑letter codes,
        with unknown/selenocysteine (‘U’, ‘O’) mapped to ‘X’.
        """
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned
        

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute mean, std, and num_class on the training set,
        then build an MLP head per layer.
        """
        # Determine whether we are working at the node, atom, or residue level
        self.view = getattr(train_set[0]["graph"], "view", "atom")

        # Collect all target values from the train set for statistics
        values_list = []
        for data in train_set:
            values_list.append(data["graph"].target)  # shape: (num_nodes,) or (num_residues,)

        values = torch.cat(values_list, dim=0)
        mean = values.float().mean()
        std = values.float().std()

        # Figure out number of classes if doing classification
        num_class = 1
        if values.dtype == torch.long:
            # If max label is >1 or not using BCE, it means multiclass
            nmax = values.max().item()
            if nmax > 1 or "bce" not in self.criterion:
                nmax += 1
            num_class = nmax

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

    def predict(self, batch, all_loss=None, metric=None):
        """
        Forward the batch through ProtALBERT up to the layer specified by the
        env-var LAYER.  Return per-residue logits from that layer only.

        Returns
        -------
        dict
            {"pred": final_logits}
        """
        # ---------------------------------------------------------------
        # 0) set-up
        # ---------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()
        self.mlp.to(device).eval()

        # ---------------------------------------------------------------
        # 1) graphs → raw sequences
        # ---------------------------------------------------------------
        graphs = batch["graph"]
        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]

        # ---------------------------------------------------------------
        # 2) tokenise with the ProtALBERT tokenizer
        # ---------------------------------------------------------------
        if not hasattr(self, "_tokenizer"):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_albert",
                do_lower_case=False,
                use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache"),
            )

        enc = self._tokenizer(
            self.prep_protalbert(sequences),          # helper that adds spaces + X for U/O
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=552,                           # ProtALBERT’s practical limit
            return_tensors="pt",
        ).to(device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]

        # ---------------------------------------------------------------
        # 3) embeddings + ALBERT-style mask
        # ---------------------------------------------------------------
        hs = self.model.embeddings(input_ids)                         # (B,L,E_word)
        hs = self.model.encoder.embedding_hidden_mapping_in(hs)       # (B,L,H)
        ext_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0     # (B,1,1,L)

        # ---------------------------------------------------------------
        # 4) figure out which layer to stop at
        # ---------------------------------------------------------------
        n_layers   = self.model.config.num_hidden_layers
        env_val    = os.getenv("LAYER", str(n_layers - 1))            # default = last layer
        last_layer = int(float(env_val))                              # allow "3" or "3.0"
        if not (0 <= last_layer < n_layers):
            raise ValueError(f"LAYER={last_layer} out of range (0-{n_layers-1})")

        # ---------------------------------------------------------------
        # 5) forward only up to that layer
        # ---------------------------------------------------------------
        groups         = self.model.encoder.albert_layer_groups
        layers_per_grp = n_layers // self.model.config.num_hidden_groups
        lidx = 0                                                     # global layer index

        with torch.no_grad():
            for grp in groups:                                       # iterate groups
                for _ in range(layers_per_grp):                      # repeat within group
                    hs = grp(
                        hs,
                        attention_mask=ext_mask,
                        head_mask=[None] * n_layers,
                        output_attentions=False,
                    )[0]
                    if lidx == last_layer:
                        break
                    lidx += 1
                if lidx == last_layer:
                    break

        # ---------------------------------------------------------------
        # 6) per-sample logits via that layer’s MLP head
        # ---------------------------------------------------------------
        final_logits = []
        for b, seq in enumerate(sequences):
            seq_len = len(seq)
            h_i   = hs[b, 1 : seq_len + 1, :]        # drop [CLS]
            log_i = self.mlp[last_layer](h_i)        # (L_res, C)
            final_logits.append(log_i)

        # ---------------------------------------------------------------
        # 7) return only what you asked for
        # ---------------------------------------------------------------
        return {"pred": final_logits}
    
    def target(self, batch):
        """
        Return a dictionary with:
          "label": the node-level target
          "mask": a boolean mask indicating which nodes are labeled
          "size": used for some metrics requiring per-graph aggregates
        """
        graph = batch["graph"]
        size = graph.num_nodes if self.view in ["node", "atom"] else graph.num_residues
        return {
            "label": graph.target,   # shape: (num_nodes,) or (num_residues,)
            "mask": graph.mask,      # shape: (num_nodes,) or (num_residues,)
            "size": size
        }

    def evaluate(self, preds, target):
        """
        Evaluate each layer's predictions given the `target`.
        preds: list of tensors, each is shape (N, num_class) or (N,) depending on your MLP output
        target: dict with { "label", "mask", "size" }
        """
        metric = {}
        _target = target["label"]
        _mask = target["mask"]
        labeled = ~torch.isnan(_target) & _mask
        _size = functional.variadic_sum(labeled.long(), target["size"])
        pred = preds["pred"]
        device = pred.device if hasattr(pred, 'device') else (
        pred["pred"].device if isinstance(pred, dict) and "pred" in pred else 
        (pred[0].device if isinstance(pred, list) and len(pred) > 0 else torch.device("cpu"))
        )

        # Move target to the same device as pred
        _target = _target.to(device)
        labeled = labeled.to(device)
        _size = _size.to(device)

        for _metric in self.metric:
            if _metric in ["mae", "rmse"]:
                # Typically for regression
                if _metric == "mae":
                    score = F.l1_loss(pred, _target, reduction="none")
                else:  # rmse
                    score = F.mse_loss(pred, _target, reduction="none").sqrt()

                score = functional.masked_mean(score, labeled, dim=0)

            elif _metric in ["micro_auroc", "micro_auprc"]:
                # Single "micro" approach across all labeled nodes
                if _metric == "micro_auroc":
                    score = metrics.area_under_roc(pred[labeled], _target[labeled])
                else:
                    score = metrics.area_under_prc(pred[labeled], _target[labeled])

            elif _metric in ["macro_auroc", "macro_auprc"]:
                # "macro" means compute per-graph, then average
                if _metric == "macro_auroc":
                    score = metrics.variadic_area_under_roc(pred[labeled], _target[labeled], _size).mean()
                else:
                    score = metrics.variadic_area_under_prc(pred[labeled], _target[labeled], _size).mean()

            elif _metric == "macro_acc":
                # One typical approach for multi-class:
                # (pred[labeled].argmax(-1) == _target[labeled]).float()
                #print(f"labeled.shape {labeled.shape}")
                #print(f"target shape {_target.shape}")
                #pred = torch.cat(pred, dim=0)
                pred = torch.cat(pred, dim=0)
                #print(f"pred.shape {pred.shape}")
                pred_argmax = pred[labeled].argmax(dim=-1)
                correct = (pred_argmax == _target[labeled]).float()
                score = functional.variadic_mean(correct, _size).mean()
                metric["macro_acc"] = score.item()

            else:
                raise ValueError(f"Unknown metric `{_metric}`")

        return metric

########################################!~!~!~!
# Exit

@R.register("tasks.EarlyExitClassificationTemperature_Node_continuous_ProtAlbert")
class EarlyExitClassificationTemperature_Node_continuous_ProtAlbert(tasks.Task, core.Configurable):
    _option_members = {"criterion", "metric"}

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("macro_auprc", "macro_auroc"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        verbose=0,
    ):
        super(EarlyExitClassificationTemperature_Node_continuous_ProtAlbert, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, disable normalization
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose


    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    @staticmethod
    def prep_protalbert(seqs):
        """
        ProtAlbert expects space‑separated, upper‑case single‑letter codes,
        with unknown/selenocysteine (‘U’, ‘O’) mapped to ‘X’.
        """
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned
        

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute mean, std, and num_class on the training set,
        then build an MLP head per layer.
        """
        # Determine whether we are working at the node, atom, or residue level
        self.view = getattr(train_set[0]["graph"], "view", "atom")

        # Collect all target values from the train set for statistics
        values_list = []
        for data in train_set:
            values_list.append(data["graph"].target)  # shape: (num_nodes,) or (num_residues,)

        values = torch.cat(values_list, dim=0)
        mean = values.float().mean()
        std = values.float().std()

        # Figure out number of classes if doing classification
        num_class = 1
        if values.dtype == torch.long:
            # If max label is >1 or not using BCE, it means multiclass
            nmax = values.max().item()
            if nmax > 1 or "bce" not in self.criterion:
                nmax += 1
            num_class = nmax

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

    def predict(self, batch, all_loss=None, metric=None):
        """
        Same inputs/outputs as the original ESM-2 version, but revised to use
        ProtALBERT.  The overall control-flow, early-exit logic, and return
        structure are unchanged so downstream code keeps working.
        """
        # ---------------------------------------------------------------
        # 0) bookkeeping & set-up
        # ---------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()
        self.mlp.to(device).eval()

        graphs = batch["graph"]
        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]
        B = len(sequences)

        # ---------------------------------------------------------------
        # 1) temperatures / thresholds
        # ---------------------------------------------------------------
        threshold = float(os.getenv("CFG_THRESHOLD"))
        percent   = float(os.getenv("PERCENT"))

        tmp_file = os.getenv("TEMPERATURE_FILE")
        if tmp_file and tmp_file.lower() != "none":
            temps = torch.tensor(self.extract_temperatures(tmp_file), device=device)
        else:
            n_layers = getattr(self, "num_layers", self.model.config.num_hidden_layers)
            temps = torch.ones(n_layers, device=device)

        # ---------------------------------------------------------------
        # 2) tokenize with ProtALBERT tokenizer
        # ---------------------------------------------------------------
        if not hasattr(self, "_tokenizer"):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_albert",           # <-- CHANGED
                do_lower_case=False, use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache"),
            )

        enc = self._tokenizer(
            self.prep_protalbert(sequences),    # <-- CHANGED
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=552,                      # ProtAlbert limit 512-550  <-- CHANGED
            return_tensors="pt",
        ).to(device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]  # (B,L)

        # ---------------------------------------------------------------
        # 3) initial embeddings (B,L,H) and Albert-style attention mask
        # ---------------------------------------------------------------
        hs = self.model.embeddings(input_ids)                         # (B,L,E_word)
        hs = self.model.encoder.embedding_hidden_mapping_in(hs)       # (B,L,H)  <-- CHANGED
        ext_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0     # (B,1,1,L)

        # ---------------------------------------------------------------
        # 4) placeholders for results & trackers (match original output)
        # ---------------------------------------------------------------
        final_logits = [None] * B
        final_layers = torch.full((B,), -1, device=device)
        best_logits  = [None] * B
        best_prob    = torch.full((B,), -float("inf"), device=device)
        best_layers  = torch.full((B,), -1, device=device)
        computed_layers = torch.full((B,), -1, device=device) 
        active       = torch.arange(B, device=device)

        n_layers        = getattr(self, "num_layers", self.model.config.num_hidden_layers)
        groups          = self.model.encoder.albert_layer_groups        # <-- CHANGED
        layers_per_grp  = n_layers // self.model.config.num_hidden_groups  # <-- CHANGED

        # ---------------------------------------------------------------
        # 5) iterate over ProtALBERT logical encoder layers
        # ---------------------------------------------------------------
        lidx = 0                                                       # global layer idx
        for grp in groups:                                             # <-- CHANGED
            for _ in range(layers_per_grp):                            # <-- CHANGED
                if active.numel() == 0:
                    break
                for idx in active.tolist():
                    computed_layers[idx] = lidx

                # ---- forward pass ONLY for active samples ------------
                hs_act = grp(                                          # <-- CHANGED
                    hs[active],
                    attention_mask=ext_mask[active],
                    head_mask=[None] * n_layers,
                    output_attentions=False,
                )[0]
                hs[active] = hs_act

                # ---- MLP head per sample ----------------------------
                logits_list  = []
                max_prob_vec = torch.empty(active.size(0), device=device)

                for loc, gidx in enumerate(active.tolist()):
                    seq_len = len(sequences[gidx])
                    h_i     = hs_act[loc, 1 : seq_len + 1, :]
                    log_i   = self.mlp[lidx](h_i)                      # <-- still indexed by lidx
                    logits_list.append(log_i)

                    probs = torch.sigmoid(log_i / temps[lidx])
                    max_prob_vec[loc] = probs.max(dim=1).values.mean()

                    if (probs.max(dim=1).values > threshold).float().mean() >= percent:
                        final_logits[gidx] = log_i
                        final_layers[gidx] = lidx

                # ---- split finished vs. still active ----------------
                done_mask    = final_layers[active] != -1
                newly_done   = active[done_mask]
                still_active = active[~done_mask]
                is_final     = lidx == n_layers - 1

                if still_active.numel() > 0:
                    better = max_prob_vec[~done_mask] > best_prob[still_active]
                    if is_final and os.getenv("SELECT_LAST", "False") == "True":
                        better = torch.ones_like(better, dtype=torch.bool)

                    if better.any():
                        upd_idx = still_active[better]
                        best_prob[upd_idx]   = max_prob_vec[~done_mask][better]
                        best_layers[upd_idx] = lidx
                        for k, g in enumerate(upd_idx.tolist()):
                            best_logits[g] = logits_list[(~done_mask).nonzero(as_tuple=True)[0][k]]

                active = still_active
                lidx  += 1                                             # advance global layer idx
            if active.numel() == 0:
                break

        # ---------------------------------------------------------------
        # 6) force-exit any leftovers -----------------------------------
        for g in active.tolist():
            final_logits[g] = best_logits[g]
            final_layers[g] = best_layers[g]

        # ---------------------------------------------------------------
        # 7) legacy ASCII-matrix (2000-wide) ----------------------------
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in sequences],
            batch_first=True,
            padding_value=0,
        )
        if ascii_mat.size(1) < 2000:
            pad = ascii_mat.new_zeros(ascii_mat.size(0), 2000 - ascii_mat.size(1))
            ascii_mat = torch.cat([ascii_mat, pad], dim=1)

        return {
            "pred": final_logits,   # list(Tensor[L_i,C])
            "layers": final_layers, # Tensor[B]
            "sequences": ascii_mat, # Tensor[B,2000]
            "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64)
        }
    def target(self, batch):
        """
        Return a dictionary with:
          "label": the node-level target
          "mask": a boolean mask indicating which nodes are labeled
          "size": used for some metrics requiring per-graph aggregates
        """
        graph = batch["graph"]
        size = graph.num_nodes if self.view in ["node", "atom"] else graph.num_residues
        return {
            "label": graph.target,   # shape: (num_nodes,) or (num_residues,)
            "mask": graph.mask,      # shape: (num_nodes,) or (num_residues,)
            "size": size
        }

    def evaluate(self, preds, target):
        """
        Evaluate each layer's predictions given the `target`.
        preds: list of tensors, each is shape (N, num_class) or (N,) depending on your MLP output
        target: dict with { "label", "mask", "size" }
        """
        metric = {}
        _target = target["label"]
        _mask = target["mask"]
        labeled = ~torch.isnan(_target) & _mask
        _size = functional.variadic_sum(labeled.long(), target["size"])
        pred = preds["pred"]

        layers = preds["layers"]
        layer_frequencies = torch.bincount(layers)
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total

        for _metric in self.metric:
            if _metric in ["mae", "rmse"]:
                # Typically for regression
                if _metric == "mae":
                    score = F.l1_loss(pred, _target, reduction="none")
                else:  # rmse
                    score = F.mse_loss(pred, _target, reduction="none").sqrt()

                score = functional.masked_mean(score, labeled, dim=0)

            elif _metric in ["micro_auroc", "micro_auprc"]:
                # Single "micro" approach across all labeled nodes
                if _metric == "micro_auroc":
                    score = metrics.area_under_roc(pred[labeled], _target[labeled])
                else:
                    score = metrics.area_under_prc(pred[labeled], _target[labeled])

            elif _metric in ["macro_auroc", "macro_auprc"]:
                # "macro" means compute per-graph, then average
                if _metric == "macro_auroc":
                    score = metrics.variadic_area_under_roc(pred[labeled], _target[labeled], _size).mean()
                else:
                    score = metrics.variadic_area_under_prc(pred[labeled], _target[labeled], _size).mean()

            elif _metric == "macro_acc":
                # One typical approach for multi-class:
                # (pred[labeled].argmax(-1) == _target[labeled]).float()
                #print(f"labeled.shape {labeled.shape}")
                #print(f"target shape {_target.shape}")
                #pred = torch.cat(pred, dim=0)
                pred = torch.cat(pred, dim=0)
                #print(f"pred.shape {pred.shape}")
                pred_argmax = pred[labeled].argmax(dim=-1)
                correct = (pred_argmax == _target[labeled]).float()
                score = functional.variadic_mean(correct, _size).mean()
                metric["macro_acc"] = score.item()

            else:
                raise ValueError(f"Unknown metric `{_metric}`")
            metric["layer"] = average_layer.item()
            computed_layers = preds["computed_layers"]
            computed_layer_frequencies = torch.bincount(computed_layers)
            total_computed = computed_layer_frequencies.sum()
            computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
            average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed
            metric["avg_computed_layer"] = average_computed_layer.item()

        return metric

########################################!~!~!~!
# Analysis

########################################
########################################
########################################
#### ESM #####

########################################!~!~!~!
########################################
## Classification

########################################!~!~!~!
# Normal

@R.register("tasks.NormalClassification_walltime")
class NormalClassification_walltime(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('auprc@micro', 'f1_max'), verbose=0, num_class=1, weight=None, confidence_threshold=None):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer.
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(NormalClassification_walltime, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.confidence_threshold = confidence_threshold
        self.metric = metric

    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures


    def predict(self, batch, all_loss=None, metric=None):

        device = self.device
        graphs = batch["graph"]

        # Convert graphs -> sequences
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            seq = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(seq)

        batch_size = len(sequences)

        # We'll store final chosen logits + chosen layer
        final_logits = [None] * batch_size
        final_layers = [None] * batch_size

        # 1) Tokenize once
        data_ = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(data_)
        batch_tokens = batch_tokens.to(device)

        # 2) Build a padding mask
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)
        # If your sequences are right-padded, we can pass `padding_mask` to the transformer

        # 3) **Replicate ESM2's forward logic** EXACTLY

        # 3a) Embedding scale
        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)

        # 3b) Token dropout, if ESM2 is using it
        # (Check self.model.model.token_dropout)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)

            # ESM2 also does a ratio-based rescaling
            # See the official code block that looks like:
            #    mask_ratio_train = 0.15 * 0.8
            #    src_lengths = (~padding_mask).sum(-1)
            #    mask_ratio_observed = (tokens == mask_idx).sum(-1).to(x.dtype) / src_lengths
            #    x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
            mask_ratio_train = 0.12  # example
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_lengths
            # avoid divide-by-zero for any empty sequences
            mask_ratio_observed = torch.clamp(mask_ratio_observed, min=1e-9)
            scale_factor = (1 - mask_ratio_train) / (1 - mask_ratio_observed)
            x = x * scale_factor.unsqueeze(-1).unsqueeze(-1)

        # 3c) If token != padding, multiply by 1 - padding_mask, etc.
        if padding_mask.any():
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # 3d) Now we do the same as ESM2: x => (T, B, E)
        x = x.transpose(0, 1)  # shape: [seq_len, batch_size, hidden_dim]

        # 4) Keep track of "active" sample indices
        active_indices = torch.arange(batch_size, device=device)

        # 5) Temperatures
        #threshold = float(os.getenv("THRESHOLD"))
        #temperature_file = str(os.getenv("TEMPERATURE_FILE"))
        #temperatures = self.extract_temperatures(temperature_file)
        #temperatures = torch.tensor(temperatures, device=device)

        # 6) Iterate over each layer exactly once
        for layer_idx, layer_module in enumerate(self.model.model.layers):
            #if len(active_indices) == 0:
            #    break

            # Gather x for active only
            # x shape is [seq_len, batch_size, hidden_dim].
            # We want to slice out "batch_size" dimension = active_indices
            # We'll do an index_select along dim=1:
            hs_active = x[:, active_indices, :]

            # apply the layer
            layer_out = layer_module(
                hs_active,
                self_attn_padding_mask=padding_mask[active_indices]
                    if padding_mask is not None else None,
                need_head_weights=False
            )
            # Some ESM versions return (hidden, attn), some just hidden
            if isinstance(layer_out, tuple):
                hs_active = layer_out[0]
            else:
                hs_active = layer_out

            # Place updated states back
            x[:, active_indices, :] = hs_active

            # Check if this is the **final** layer
            is_final_layer = (layer_idx == self.model.model.num_layers - 1)
            if is_final_layer:
                # ESM2 does a final layer norm after the loop
                hs_active = self.model.model.emb_layer_norm_after(hs_active)

            # We want to feed the representation to an MLP
            # Usually ESM2 store "representations[layer_idx+1]" as hs_active.transpose(0,1)
            # But let's just do the same for MLP
            hs_for_mlp = hs_active.transpose(0, 1)  # => [num_active, seq_len, hidden_dim]

            # If normal classification used mean-pooling for each sample, do it here:
            mlp_input = hs_for_mlp.mean(dim=1)  # shape [num_active, hidden_dim]

            # Then apply the layer's MLP
            logits_active = self.mlp[layer_idx](mlp_input)

            # Apply temperature scaling & threshold
            #scaled_logits = logits_active / temperatures[layer_idx]
            #probabilities = torch.sigmoid(scaled_logits)
            #max_prob, _ = probabilities.view(probabilities.size(0), -1).max(dim=1)
            #meet_threshold_mask = (max_prob > threshold)

            #newly_exited = active_indices[meet_threshold_mask]
            #still_active = active_indices[~meet_threshold_mask]

            # Save final logits/layer for those who exit
            #for i, global_idx in enumerate(newly_exited.tolist()):
            #    final_logits[global_idx] = logits_active[meet_threshold_mask][i]
            #    final_layers[global_idx] = layer_idx

            # Update active_indices
            #active_indices = still_active

        # 7) If any remain after the final layer, they're forced to exit
        if len(active_indices) > 0:
            # we already computed final layer above (with LN),
            # so let's apply the final MLP again for them, if needed
            last_layer_idx = self.model.model.num_layers - 1
            for i, global_idx in enumerate(active_indices.tolist()):
                # Hidden states for that sample
                # shape: [seq_len, hidden_dim]
                final_h = x[:, global_idx, :]
                # final LN is presumably done
                final_h = self.model.model.emb_layer_norm_after(final_h)
                final_h = final_h.unsqueeze(1).transpose(0, 1)  # => [1, seq_len, hidden_dim]
                mlp_input = final_h.mean(dim=1)  # => [1, hidden_dim]
                logits_final = self.mlp[last_layer_idx](mlp_input)[0]
                final_logits[global_idx] = logits_final
                final_layers[global_idx] = last_layer_idx

        # 8) Stack results
        selected_outputs = torch.stack(final_logits, dim=0)

        encoded_sequences = []
        max_len = 2000
        for seq in sequences:
            ascii_ids = [ord(c) for c in seq]
            padded = ascii_ids + [0] * (max_len - len(seq))
            encoded_sequences.append(padded)

        return {"pred":selected_outputs, "layers":torch.tensor(final_layers, device=self.device, dtype=torch.int64), "sequences":torch.tensor(encoded_sequences, device=self.device, dtype=torch.int64)} 

    def predict_old(self, batch, all_loss=None, metric=None):
        threshold = float(os.getenv("THRESHOLD"))
        device = self.device
        graphs = batch["graph"]
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)
        batch_size = len(sequences)
        final_logits = [None]* batch_size
        final_layers = [None] * batch_size
        input = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(input)
        batch_tokens = batch_tokens.to(device)
        
        #embed tokens
        hidden_states = self.model.model.embed_tokens(batch_tokens)
        if hasattr(self.model.model, "embed_positions") and self.model.model.embed_positions is not None:
            hidden_states = hidden_states + self.model.model.embed_positions(batch_tokens)
        if hasattr(self.model.model, "dropout_module"):
            hidden_states = self.model.model.dropout_module(hidden_states)

        #track active samples
        active_indices = torch.arange(batch_size, device=device)

        #load temperature values 
        temperature_file = str(os.getenv("TEMPERATURE_FILE"))
        temperatures = self.extract_temperatures(temperature_file)
        temperatures = torch.tensor(temperatures, device=self.device)

        #iterate over esm2 layers
        for layer_idx, layer_module in enumerate(self.model.model.layers):
            if len(active_indices) == 0:
                break
            hs_active = hidden_states[active_indices]
            layer_out = layer_module(hs_active)
            if isinstance(layer_out, tuple):
                hs_active = layer_out[0]
            else:
                hs_active = layer_out
            hidden_states[active_indices] = hs_active 
            mlp_input = hs_active.mean(dim=1)
            #logits_active = self.mlp[layer_idx](hs_active)
            logits_active = self.mlp[layer_idx](mlp_input)
            scaled_logits = logits_active / temperatures[layer_idx]
            probabilities = torch.sigmoid(scaled_logits)
            max_prob, _ = probabilities.view(probabilities.size(0), -1).max(dim=1)
            meet_threshold_mask = (max_prob > threshold)
            newly_exited = active_indices[meet_threshold_mask]
            still_active = active_indices[~meet_threshold_mask]
            for i, global_idx in enumerate(newly_exited.tolist()):
                final_logits[global_idx] = logits_active[meet_threshold_mask][i]
                final_layers[global_idx] = layer_idx
            active_indices = still_active
        if len(active_indices) > 0:
            last_layer_idx = self.model.num_layers - 1
            for i, global_idx in enumerate(active_indices.tolist()):
                # The hidden states are already from the final layer now
                final_logits[global_idx] = self.mlp[last_layer_idx](hidden_states[global_idx].unsqueeze(0))[0]
                final_layers[global_idx] = last_layer_idx   
        selected_outputs = torch.stack(final_logits, dim=0)
        selected_layers = torch.tensor(final_layers, device=device, dtype=torch.int64)
        encoded_sequences = []
        max_len = 2000
        for seq in sequences:
            ascii_ids = [ord(c) for c in seq]
            padded = ascii_ids + [0] * (max_len - len(seq))
            encoded_sequences.append(padded)
        return {"pred":selected_outputs, "layers":torch.tensor(selected_layers, device=self.device, dtype=torch.int64), "sequences":torch.tensor(encoded_sequences, device=self.device, dtype=torch.int64)} 
    
    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        pred = preds["pred"]
        layers = preds["layers"]
        sequences = preds["sequences"]
        metric = {}
        layer_frequencies = torch.bincount(layers)
        for layer_idx, freq in enumerate(layer_frequencies):
            print(f"Layer {layer_idx}: Frequency {freq.item()}")
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)
            name = tasks._get_metric_name(_metric)
            metric[name] = score
        results = {"preds": pred, "target": target, "layers": layers, "metric": metric, "sequences": sequences}
        results_file = os.getenv("RESULT_FILE")
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        return metric


########################################!~!~!~!
# Exit


@R.register("tasks.EarlyExitClassification_walltime")
class EarlyExitClassification_walltime(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('auprc@micro', 'f1_max'), verbose=0, num_class=1, weight=None, confidence_threshold=None):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer.
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(EarlyExitClassification_walltime, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.confidence_threshold = confidence_threshold
        self.metric = metric

    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures


    def predict(self, batch, all_loss=None, metric=None):

        device = self.device
        graphs = batch["graph"]

        # Convert graphs -> sequences
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            seq = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(seq)

        batch_size = len(sequences)

        # We'll store final chosen logits + chosen layer
        final_logits = [None] * batch_size
        final_layers = [None] * batch_size
        best_prob     = torch.full((batch_size,), -float("inf"), device=device)   # NEW
        best_logits   = [None] * batch_size                                       # NEW
        best_layers   = [None] * batch_size
        computed_layers = [None] * batch_size                                       

        # 1) Tokenize once
        data_ = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(data_)
        batch_tokens = batch_tokens.to(device)

        # 2) Build a padding mask
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)
        # If your sequences are right-padded, we can pass `padding_mask` to the transformer

        # 3) **Replicate ESM2's forward logic** EXACTLY

        # 3a) Embedding scale
        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)

        # 3b) Token dropout, if ESM2 is using it
        # (Check self.model.model.token_dropout)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)

            # ESM2 also does a ratio-based rescaling
            # See the official code block that looks like:
            #    mask_ratio_train = 0.15 * 0.8
            #    src_lengths = (~padding_mask).sum(-1)
            #    mask_ratio_observed = (tokens == mask_idx).sum(-1).to(x.dtype) / src_lengths
            #    x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
            mask_ratio_train = 0.12  # example
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_lengths
            # avoid divide-by-zero for any empty sequences
            mask_ratio_observed = torch.clamp(mask_ratio_observed, min=1e-9)
            scale_factor = (1 - mask_ratio_train) / (1 - mask_ratio_observed)
            x = x * scale_factor.unsqueeze(-1).unsqueeze(-1)

        # 3c) If token != padding, multiply by 1 - padding_mask, etc.
        if padding_mask.any():
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # 3d) Now we do the same as ESM2: x => (T, B, E)
        x = x.transpose(0, 1)  # shape: [seq_len, batch_size, hidden_dim]

        # 4) Keep track of "active" sample indices
        active_indices = torch.arange(batch_size, device=device)

        # 5) Temperatures
        threshold = float(os.getenv("THRESHOLD"))
        temperature_file = os.getenv("TEMPERATURE_FILE")
        if temperature_file is not None and temperature_file != 'None':
            temperatures = self.extract_temperatures(temperature_file)
            temperatures = torch.tensor(temperatures, device=device)
        else:
           temperatures = torch.ones(33)


        # 6) Iterate over each layer exactly once
        for layer_idx, layer_module in enumerate(self.model.model.layers):
            if len(active_indices) == 0:
                break
            for idx in active_indices.tolist():
                computed_layers[idx] = layer_idx

            # Gather x for active only
            # x shape is [seq_len, batch_size, hidden_dim].
            # We want to slice out "batch_size" dimension = active_indices
            # We'll do an index_select along dim=1:
            hs_active = x[:, active_indices, :]

            # apply the layer
            layer_out = layer_module(
                hs_active,
                self_attn_padding_mask=padding_mask[active_indices]
                    if padding_mask is not None else None,
                need_head_weights=False
            )
            # Some ESM versions return (hidden, attn), some just hidden
            if isinstance(layer_out, tuple):
                hs_active = layer_out[0]
            else:
                hs_active = layer_out

            # Place updated states back
            x[:, active_indices, :] = hs_active

            # Check if this is the **final** layer
            is_final_layer = (layer_idx == self.model.model.num_layers - 1)
            if is_final_layer:
                # ESM2 does a final layer norm after the loop
                hs_active = self.model.model.emb_layer_norm_after(hs_active)

            # We want to feed the representation to an MLP
            # Usually ESM2 store "representations[layer_idx+1]" as hs_active.transpose(0,1)
            # But let's just do the same for MLP
            hs_for_mlp = hs_active.transpose(0, 1)  # => [num_active, seq_len, hidden_dim]

            # If normal classification used mean-pooling for each sample, do it here:
            mlp_input = hs_for_mlp.mean(dim=1)  # shape [num_active, hidden_dim]

            # Then apply the layer's MLP
            logits_active = self.mlp[layer_idx](mlp_input)

            # Apply temperature scaling & threshold
            scaled_logits = logits_active / temperatures[layer_idx]
            probabilities = torch.sigmoid(scaled_logits)
            max_prob, _ = probabilities.view(probabilities.size(0), -1).max(dim=1)

            # ---------- NEW: keep the best prob/logits seen so far ----------
            better = max_prob > best_prob[active_indices]
            if is_final_layer and os.environ.get("SELECT_LAST", "False") == "True":
                better = torch.full_like(best_prob[active_indices], fill_value=True, dtype=torch.bool)

            if better.any():
                idx_global = active_indices[better]          # indices in the original batch
                best_prob[idx_global]   = max_prob[better]   # update tensor (in‑place assignment)
                for g in idx_global.tolist():                # ✅ update Python lists
                    best_layers[g] = layer_idx
                # store **raw** logits (same as you already return when a sample exits)
                best_logits_arr = logits_active[better]      # shape [n_better, num_classes]
                for j, g in enumerate(idx_global.tolist()):
                    best_logits[g] = best_logits_arr[j]
            # ----------------------------------------



            meet_threshold_mask = (max_prob > threshold)

            newly_exited = active_indices[meet_threshold_mask]
            still_active = active_indices[~meet_threshold_mask]

            # Save final logits/layer for those who exit
            for i, global_idx in enumerate(newly_exited.tolist()):
                final_logits[global_idx] = logits_active[meet_threshold_mask][i]
                final_layers[global_idx] = layer_idx

            # Update active_indices
            active_indices = still_active

        # 7) If any remain after the final layer, they're forced to exit
        if len(active_indices) > 0:
            # we already computed final layer above (with LN),
            # so let's apply the final MLP again for them, if needed
            for g in active_indices.tolist():
                final_logits[g] = best_logits[g]
                final_layers[g] = best_layers[g]


        # 8) Stack results
        selected_outputs = torch.stack(final_logits, dim=0)

        encoded_sequences = []
        max_len = 2000
        for seq in sequences:
            ascii_ids = [ord(c) for c in seq]
            padded = ascii_ids + [0] * (max_len - len(seq))
            encoded_sequences.append(padded)

        return {"pred":selected_outputs, "layers":torch.tensor(final_layers, device=self.device, dtype=torch.int64), "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64), "sequences":torch.tensor(encoded_sequences, device=self.device, dtype=torch.int64)} 

    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        result = {}
        pred = preds["pred"]
        layers = preds["layers"]
        computed_layers = preds["computed_layers"]
        sequences = preds["sequences"]
        metric = {}
        layer_frequencies = torch.bincount(layers)
        computed_layer_frequencies = torch.bincount(computed_layers)
        f1_max = None
        for layer_idx, freq in enumerate(layer_frequencies):
            print(f"Layer {layer_idx}: Frequency {freq.item()}")
        
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
                f1_max = score
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)
            name = tasks._get_metric_name(_metric)
            metric[name] = score
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total

        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed

        if os.getenv("RESULT_FILE"):
            results_file = os.getenv("RESULT_FILE")
            with open(results_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([os.getenv("THRESHOLD"), f1_max.item(), average_layer.item(), average_computed_layer.item()])
        elif os.getenv("RESULT_PICKLE"):
            results_pickle = os.getenv("RESULT_PICKLE")
            results = {"preds": pred, "target": target, "layers": layers, "avg_computed_layer": average_computed_layer, "metric": metric, "sequences": sequences}
            with open(results_pickle, 'wb') as f:
                pickle.dump(results, f)
        result["f1"] = f1_max.item()
        result["avg_layer"] = average_layer.item()
        result["avg_computed_layer"] = average_computed_layer.item()
        return result
    

########################################!~!~!~!
# Analysis


import math
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
@R.register("tasks.EarlyExitClassification_walltime_analysis")
class EarlyExitClassification_walltime_analysis(tasks.Task, core.Configurable):
    """
    • Runs every sample through *all* transformer layers.
    • Stores the full per-class probability vector for each layer.
    • During evaluation, finds the probability threshold τ that maximises
      micro-F1 on the *last* layer (exactly what `metrics.f1_max` does), then
      applies that τ to every layer to decide if the sample is correct.
    """

    eps = 1e-10
    _option_members = {"task", "metric"}

    # ------------------------------------------------------------------ #
    # 1) Constructor / helpers                                           #
    # ------------------------------------------------------------------ #
    def __init__(self,
                 model,
                 task=(),
                 metric=("auprc@micro", "f1_max"),
                 num_mlp_layer=2,
                 confidence_threshold=None,
                 verbose=0,
                 num_class=1,
                 weight=None):

        super(EarlyExitClassification_walltime_analysis, self).__init__()
        self.model     = model
        self.task      = task
        self.metric    = metric
        self.num_layers = model.num_layers
        self.verbose   = verbose

        # Freeze backbone weights – analysis only.
        for p in self.model.parameters():
            p.requires_grad = False

        # Make sure we can access the per-layer MLP heads:
        #   • If they are already attached to `model` (usual case), just use them.
        #   • Otherwise build a minimal set (weight-less) so code runs without error.

    # ---- helper to read per-layer temperatures ------------------------ #
    def extract_temperatures(self, file_path):
        temps = []
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                temps.append(float(row[1].split("(")[1].split(",")[0]))
        return temps

    # ------------------------------------------------------------------ #
    # 2) Predict – run forward pass and record all probabilities          #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, batch, all_loss=None, metric=None):
        device  = self.device
        graphs  = batch["graph"]

        # --- graphs → AA sequences ------------------------------------- #
        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]
        B = len(sequences)

        # --- tokenise --------------------------------------------------- #
        inp = [(f"protein_{i}", s) for i, s in enumerate(sequences)]
        _, _, toks = self.model.alphabet.get_batch_converter()(inp)
        toks    = toks.to(device)
        padding = toks.eq(self.model.model.padding_idx)

        # --- temperatures ---------------------------------------------- #
        temp_fp = os.getenv("TEMPERATURE_FILE")
        if temp_fp and temp_fp.lower() != "none":
            temps = torch.tensor(self.extract_temperatures(temp_fp), device=device)
        else:
            temps = torch.ones(self.num_layers, device=device)

        # --- forward through layers ------------------------------------ #
        x = self.model.model.embed_scale * self.model.model.embed_tokens(toks)
        if padding.any():
            x = x * (1 - padding.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)   # [T, B, E]

        # allocate cube [L, B, C] once we know C
        layer_prob_full = None

        for l_idx, layer in enumerate(self.model.model.layers):
            h = layer(
                x,
                self_attn_padding_mask=padding if padding is not None else None,
                need_head_weights=False,
            )[0]
            x = h

            if l_idx == self.model.model.num_layers - 1:
                h = self.model.model.emb_layer_norm_after(h)

            mlp_in = h.transpose(0, 1).mean(dim=1)             # [B, E]
            logits = self.mlp[l_idx](mlp_in)                   # [B, C]
            probs  = torch.sigmoid(logits / temps[l_idx])      # [B, C]

            if layer_prob_full is None:
                C = probs.size(1)
                layer_prob_full = torch.empty(self.num_layers, B, C,
                                              device=device, dtype=probs.dtype)
            layer_prob_full[l_idx] = probs

        # transpose to [B, L, C] for convenience
        layer_prob_full = layer_prob_full.permute(1, 0, 2).contiguous()

        # --- stash ASCII-encoded sequences for traceability ------------- #
        max_len = 2000
        encoded = [
            [ord(c) for c in seq] + [0] * (max_len - len(seq))
            for seq in sequences
        ]

        return {
            "sequences"       : torch.tensor(encoded, device=device, dtype=torch.int64),
            "layer_prob_full" : layer_prob_full                 # [B, L, C]
        }

    # ------------------------------------------------------------------ #
    # 3) Target                                                          #
    # ------------------------------------------------------------------ #
    def target(self, batch):
        # batch["targets"] is expected to be a multi-hot tensor [B, C]
        return batch["targets"]

    # ------------------------------------------------------------------ #
    # 4) Evaluate – F1-consistent correctness                            #
    # ------------------------------------------------------------------ #

    def _torchdrug_f1_max_with_tau(self, pred, target):
        """
        pred   : Tensor [B, C]   – probabilities (after sigmoid & temperature)
        target : Tensor [B, C]   – binary ground-truth (0/1)

        Returns
        -------
        best_f1 : float
        best_tau: float
        """
        # (verbatim TorchDrug code, but we keep the index of max F1)
        order   = pred.argsort(descending=True, dim=1)
        target_ = target.gather(1, order)

        precision = target_.cumsum(1) / torch.ones_like(target_).cumsum(1)
        recall    = target_.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)

        is_start = torch.zeros_like(target_).bool()
        is_start[:, 0] = 1
        is_start = torch.scatter(is_start, 1, order, is_start)

        all_order = pred.flatten().argsort(descending=True)
        order_f   = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
        order_f   = order_f.flatten()

        inv_order = torch.zeros_like(order_f)
        inv_order[order_f] = torch.arange(order_f.shape[0], device=order.device)

        is_start  = is_start.flatten()[all_order]
        all_order = inv_order[all_order]

        precision = precision.flatten()
        recall    = recall.flatten()

        all_precision = precision[all_order] - \
                        torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
        all_precision = all_precision.cumsum(0) / is_start.cumsum(0)

        all_recall = recall[all_order] - \
                    torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
        all_recall = all_recall.cumsum(0) / pred.shape[0]

        all_f1     = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
        all_f1 = torch.nan_to_num(all_f1, nan=-1.0)
        best_idx   = all_f1.argmax()
        best_f1    = all_f1[best_idx]



        # The threshold is the corresponding prediction value
        flat_pred_sorted = pred.flatten()[all_order]
        best_tau   = flat_pred_sorted[best_idx]

        return best_f1.item(), best_tau.item()

    def evaluate(self, preds, target):
        """
        • τ chosen to maximise micro-F1 on the last layer.
        • For every layer we compute sample-level F1.
          correct[s, l] = 1  ⇔  sample-F1 == 1
        • Also keep layer-confidence = max class-prob.
        """
        probs   = preds["layer_prob_full"]           # [B, L, C]
        seqs    = preds["sequences"]
        B, L, C = probs.shape
        tgt_bool = target.bool()                     # [B, C]

        # ---- 1) τ via TorchDrug on last layer --------------------------------
        last_probs        = probs[:, -1]             # [B, C]
        best_f1, best_tau = self._torchdrug_f1_max_with_tau(last_probs, target)

        # ---- 2) binarise all layers ------------------------------------------
        pred_bin_all = (probs >= best_tau)           # [B, L, C] (bool)

        # ---- 3) sample-level F1 per layer ------------------------------------
        gt_exp   = tgt_bool.unsqueeze(1)             # [B, 1, C] → [B, L, C] via broadcast
        tp       = (pred_bin_all & gt_exp).sum(dim=2).float()      # [B, L]
        fp       = (pred_bin_all & ~gt_exp).sum(dim=2).float()     # [B, L]
        fn       = (~pred_bin_all & gt_exp).sum(dim=2).float()     # [B, L]
        sample_f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)            # [B, L]

        correct = sample_f1                                # [B, L] bool

        # ---- 4) layer confidence (max probability) ---------------------------
        layer_conf = probs.max(dim=2).values                        # [B, L]

        # ---- 5) optional dumps -----------------------------------------------
        csv_fp = os.getenv("RESULT_CSV")
        if csv_fp:
            with open(csv_fp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    ["sample_idx", "layer_idx", "confidence", "sample_f1", "correct"]
                )
                for s in range(B):
                    for l in range(L):
                        w.writerow([
                            s,
                            l,
                            layer_conf[s, l].item(),
                            sample_f1[s, l].item(),
                            int(correct[s, l].item())
                        ])

        pkl_fp = os.getenv("RESULT_PICKLE")
        if pkl_fp:
            with open(pkl_fp, "wb") as f:
                pickle.dump(
                    {
                        "sequences"      : seqs.cpu(),
                        "layer_prob_full": probs.cpu(),
                        "layer_conf"     : layer_conf.cpu(),
                        "sample_f1"      : sample_f1.cpu(),
                        "correct"        : correct.cpu(),
                        "best_tau"       : best_tau,
                        "best_f1"        : best_f1,
                    },
                    f,
                )

        # ---- 6) return tensors ----------------------------------------------
        return {
            "layer_conf" : layer_conf.cpu(),   # float  [B, L]
            "sample_f1"  : sample_f1.cpu(),    # float  [B, L]
            "correct"    : correct.cpu(),      # bool   [B, L]
            "best_tau"   : best_tau,
            "best_f1"    : best_f1,
        }


########################################!~!~!~!
########################################
## Property

########################################!~!~!~!
# Normal

@R.register("tasks.NormalProperty_continuous")
class NormalProperty_continuous(tasks.Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super(NormalProperty_continuous, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_layers = model.num_layers
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        for param in self.model.parameters():
            param.requires_grad = False


    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures
    
    def predict(self, batch, all_loss=None, metric=None):

        device = self.device
        graphs = batch["graph"]
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)
        batch_size = len(sequences)
        input = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(input)
        batch_tokens = batch_tokens.to(device)
        
        # ---------- ESM-2 embedding (same as Classification version) ----------
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)

        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)
        # x = x + self.model.model.embed_positions(batch_tokens)
        # x = self.model.model.emb_layer_norm_before(x)

        # token-dropout (if present)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.12
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_lengths
            mask_ratio_observed = torch.clamp(mask_ratio_observed, min=1e-9)
            x = x * ((1 - mask_ratio_train) / (1 - mask_ratio_observed)).unsqueeze(-1).unsqueeze(-1)

        if padding_mask.any():
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # ESM expects [T,B,E]
        x = x.transpose(0, 1)
        # ---------------------------------------------------------------------
        logits_per_layer = []
        for layer_idx, layer_module in enumerate(self.model.model.layers):
            layer_out = layer_module(
                x,
                self_attn_padding_mask=padding_mask
                    if padding_mask is not None else None,
                need_head_weights=False,
            )
            if isinstance(layer_out, tuple):
                x = layer_out[0]
            else:
                x = layer_out

            is_final = layer_idx == self.model.model.num_layers - 1
            if is_final:
                x = self.model.model.emb_layer_norm_after(x)

            mlp_input = x.transpose(0,1)
            mlp_input = mlp_input.mean(dim=1) ##adding mean pooling MODIFICATIOn
            logits = self.mlp[layer_idx](mlp_input)
            logits_per_layer.append(logits)

        selected_outputs = torch.stack(logits_per_layer, dim=1)
        #selected_outputs = torch.stack(final_logits, dim=0)

        return {"pred":selected_outputs} 
 
    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, preds, target):
        predictions = preds["pred"]
        labeled = ~torch.isnan(target)
        metric = {}
        metrics_out = {}
        for layer_idx in range(33):
            pred = predictions[:, layer_idx, :]
            for _metric in self.metric:
                if _metric == "auroc@micro":
                    score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
                elif _metric == "auprc@micro":
                    score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
                elif _metric == "f1_max":
                    score = metrics.f1_max(pred, target)
                elif _metric == "acc":
                    score = []
                    num_class = 0
                    for i, cur_num_class in enumerate(self.num_class):
                        _pred = pred[:, num_class:num_class + cur_num_class]
                        _target = target[:, i]
                        _labeled = labeled[:, i]
                        _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                        score.append(_score)
                        num_class += cur_num_class
                    score = torch.stack(score)
                    metrics_out[f"acc@layer{layer_idx}"] = score.item()
                else:
                    raise ValueError("Unknown criterion `%s`" % _metric)
                name = tasks._get_metric_name(_metric)
                for t, s in zip(self.task, score):
                    print("stored metric: %s [%s]" % (name, t))
                    metric["%s [%s]" % (name, t)] = s
        results_file = os.getenv("RESULT_FILE")
        with open(results_file, 'a') as f:
           writer = csv.writer(f)
           for layer_idx in range(33):
            writer.writerow([layer_idx, metrics_out[f"acc@layer{layer_idx}"]])
        #results_pickle = os.getenv("RESULT_PICKLE")
        #results = {"preds": pred, "target": target, "sequences": sequences}
        # with open(results_pickle, 'wb') as f:
        #     pickle.dump(results, f)

        return metric

########################################!~!~!~!
# Exit

@R.register("tasks.EarlyExitProperty_walltime")
class EarlyExitProperty_walltime(tasks.Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super(EarlyExitProperty_walltime, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_layers = model.num_layers
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        for param in self.model.parameters():
            param.requires_grad = False


    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures
    
    def predict(self, batch, all_loss=None, metric=None):
        device = self.device
        graphs = batch["graph"]
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)
        batch_size = len(sequences)
        final_logits = [None]* batch_size
        final_layers = [None] * batch_size
        best_prob     = torch.full((batch_size,), -float("inf"), device=device)   # NEW
        #best_logits   = [None] * batch_size                                       # NEW
        best_logits = torch.empty(batch_size, 10, device=device)
        best_layers   = [None] * batch_size  
        input = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(input)
        batch_tokens = batch_tokens.to(device)
        
        # ---------- ESM-2 embedding (same as Classification version) ----------
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)

        # core = self.model.model

        #         # ---- token embedding -------------------------------------------------
        # if hasattr(core, "embed_tokens"):                 # ESM-1b style
        #     x = core.embed_scale * core.embed_tokens(batch_tokens)
        # else:                                             # ESM-2 style
        #     x = core.embed_scale * core.token_embedding(batch_tokens)

        # # ---- positional embedding -------------------------------------------
        # if hasattr(core, "embed_positions"):              # ESM-1b style
        #     x = x + core.embed_positions(batch_tokens)
        # else:                                             # ESM-2 style (learned, stored as a Parameter)
        #     pos_emb = core.position_embedding[:, : x.size(1), :]
        #     x = x + pos_emb

        # # ---- pre-LayerNorm (optional) ---------------------------------------
        # if hasattr(core, "emb_layer_norm_before"):        # ESM-1b style
        #     x = core.emb_layer_norm_before(x)
        # elif hasattr(core, "layernorm_before"):           # ESM-2 style
        #     x = core.layernorm_before(x)

        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)
        # x = x + self.model.model.embed_positions(batch_tokens)
        # x = self.model.model.emb_layer_norm_before(x)

        # token-dropout (if present)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.12
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_lengths
            mask_ratio_observed = torch.clamp(mask_ratio_observed, min=1e-9)
            x = x * ((1 - mask_ratio_train) / (1 - mask_ratio_observed)).unsqueeze(-1).unsqueeze(-1)

        if padding_mask.any():
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # ESM expects [T,B,E]
        x = x.transpose(0, 1)
        # ---------------------------------------------------------------------

        # When `threshold` is not explicitly passed, fall back to env var
        threshold = float(os.getenv("THRESHOLD"))

        # Temperatures (optional)
        temperature_file = os.getenv("TEMPERATURE_FILE")
        temperatures = torch.tensor(self.extract_temperatures(temperature_file),
                                        device=device)

        # ---------- early-exit loop (identical to Classification) -------------
        active_indices = torch.arange(batch_size, device=device)

        for layer_idx, layer_module in enumerate(self.model.model.layers):
            if len(active_indices) == 0:
                break

            hs_active = x[:, active_indices, :]
            layer_out = layer_module(
                hs_active,
                self_attn_padding_mask=padding_mask[active_indices]
                    if padding_mask is not None else None,
                need_head_weights=False,
            )
            if isinstance(layer_out, tuple):
                hs_active = layer_out[0]
            else:
                hs_active = layer_out

            x[:, active_indices, :] = hs_active

            is_final = layer_idx == self.model.model.num_layers - 1
            if is_final:
                hs_active = self.model.model.emb_layer_norm_after(hs_active)

            # hs_bt = hs_active.transpose(0, 1)
            # mask = ~padding_mask[active_indices]
            # denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            # mlp_input = (hs_bt * mask.unsqueeze(-1)).sum(dim=1) / denom
            #mlp_input = hs_active.transpose(0, 1).mean(dim=1)        # [N_active, E]
            mlp_input = hs_active.transpose(0,1)
            mlp_input = mlp_input.mean(dim=1) ##adding mean pooling MODIFICATIOn
            logits_active = self.mlp[layer_idx](mlp_input)

            scaled_logits = logits_active / temperatures[layer_idx]
            probabilities = torch.sigmoid(scaled_logits)
            max_prob, _ = probabilities.view(probabilities.size(0), -1).max(dim=1)

            # keep best prob/logits per sample
            better = max_prob > best_prob[active_indices]
            if is_final and os.environ.get("SELECT_LAST", "False") == "True":
                better = torch.full_like(best_prob[active_indices], fill_value=True, dtype=torch.bool)
            if better.any():
                idx_global = active_indices[better]          # indices in the original batch
                best_prob[idx_global]   = max_prob[better]   # update tensor (in‑place assignment)
                best_logits[idx_global] = logits_active[better]
                #best_layers[idx_global] = layer_idx
                # for g in idx_global.tolist():                # ✅ update Python lists
                #     best_layers[g] = layer_idx
                # store **raw** logits (same as you already return when a sample exits)
                # best_logits_arr = logits_active[better]      # shape [n_better, num_classes]
                # for j, g in enumerate(idx_global.tolist()):
                #     best_logits[g] = best_logits_arr[j]'
                for g in idx_global.tolist():                 # ✅ loop over plain ints
                    best_layers[g] = layer_idx

            newly_exited   = active_indices[max_prob > threshold]
            still_active   = active_indices[max_prob <= threshold]

            for i, g in enumerate(newly_exited.tolist()):
                final_logits[g] = logits_active[max_prob > threshold][i]
                final_layers[g] = layer_idx

            active_indices = still_active

        # force exit for whoever is left
        if len(active_indices) > 0:
            # we already computed final layer above (with LN),
            # so let's apply the final MLP again for them, if needed
            for g in active_indices.tolist():
                final_logits[g] = best_logits[g]
                final_layers[g] = best_layers[g]

        selected_outputs = torch.stack(final_logits, dim=0)
        #selected_outputs = torch.stack(final_logits, dim=0)

        encoded_sequences = []
        max_len = 2000
        for seq in sequences:
            ascii_ids = [ord(c) for c in seq]
            padded = ascii_ids + [0] * (max_len - len(seq))
            encoded_sequences.append(padded)

        return {"pred":selected_outputs, "layers":torch.tensor(final_layers, device=self.device, dtype=torch.int64), "sequences":torch.tensor(encoded_sequences, device=self.device, dtype=torch.int64)} 

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, preds, target):
        pred = preds["pred"]
        layers = preds["layers"]
        sequences=preds["sequences"]
        labeled = ~torch.isnan(target)
        metric = {}
        layer_frequencies = torch.bincount(layers)
        for layer_idx, freq in enumerate(layer_frequencies):
            print(f"Layer {layer_idx}: Frequency {freq.item()}")
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
                acc = score
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)
            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                print("stored metric: %s [%s]" % (name, t))
                metric["%s [%s]" % (name, t)] = s
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total
        results_file = os.getenv("RESULT_FILE")
        results = {}
        results["acc"] = acc.item()
        results["average_layer"] = average_layer.item()
        # with open(results_file, 'a') as f:
        #    writer = csv.writer(f)
        #    writer.writerow([os.getenv("THRESHOLD"), acc.item(), average_layer.item()])
        # results_pickle = os.getenv("RESULT_PICKLE")
        # results = {"preds": pred, "target": target, "layers": layers, "metric": metric, "sequences": sequences}
        # with open(results_pickle, 'wb') as f:
        #     pickle.dump(results, f)

        return results


########################################!~!~!~!
# Analysis

@R.register("tasks.EarlyExitProperty_walltime_analysis")
class EarlyExitProperty_walltime_analysis(tasks.Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super(EarlyExitProperty_walltime_analysis, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_layers = model.num_layers
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        for param in self.model.parameters():
            param.requires_grad = False


    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures
    
    def predict(self, batch, all_loss=None, metric=None):
        device = self.device
        graphs = batch["graph"]

        # -------- rebuild raw sequences -----------------------------------
        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]
        B = len(sequences)

        # tensors that will hold *per-layer* results for every sample
        layer_pred_prob   = torch.full((self.num_layers, B), float("nan"), device=device)
        layer_pred_label  = torch.full((self.num_layers, B), -1, dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # everything up to  the early-exit loop is identical to your code…
        # ------------------------------------------------------------------
        final_logits = [None] * B
        final_layers = [None] * B
        best_prob    = torch.full((B,), -float("inf"), device=device)
        best_logits  = torch.empty(B, 10, device=device)        # keep as in your code
        best_layers  = [None] * B

        input      = [(f"protein_{i}", s) for i, s in enumerate(sequences)]
        _, _, toks = self.model.alphabet.get_batch_converter()(input)
        toks       = toks.to(device)
        padding    = toks.eq(self.model.model.padding_idx)

        x = self.model.model.embed_scale * self.model.model.embed_tokens(toks)
        if padding.any():
            x = x * (1 - padding.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)        # [T, B, E]
        temperature_fp = os.getenv("TEMPERATURE_FILE")
        temps          = torch.tensor(self.extract_temperatures(temperature_fp), device=device)

        active = torch.arange(B, device=device)

        for layer_idx, layer in enumerate(self.model.model.layers):
            h_active = layer(
                x[:, active, :],
                self_attn_padding_mask=padding[active] if padding is not None else None,
                need_head_weights=False,
            )[0]            # keep the hidden-states
            x[:, active, :] = h_active             # write back
            if layer_idx == self.model.model.num_layers - 1:
                h_active = self.model.model.emb_layer_norm_after(h_active)

            mlp_in   = h_active.transpose(0, 1).mean(dim=1)      # [N_active, E]
            logits   = self.mlp[layer_idx](mlp_in)
            probs    = torch.sigmoid(logits / temps[layer_idx])   # [N_active, C]

            # --- store per-layer probability & predicted label -------------
            max_prob, pred_label = probs.max(dim=1)
            layer_pred_prob[layer_idx, active]  = max_prob
            layer_pred_label[layer_idx, active] = pred_label
            # ----------------------------------------------------------------

        # ----- pack up everything to return --------------------------------
        max_len = 2000
        encoded = [
            [ord(c) for c in seq] + [0] * (max_len - len(seq))
            for seq in sequences
        ]
        layer_pred_prob  = layer_pred_prob.t().contiguous()
        layer_pred_label = layer_pred_label.t().contiguous()
        return {
            "sequences"       : torch.tensor(encoded, device=device, dtype=torch.int64),
            # NEW ↓↓↓
            "layer_pred_prob" : layer_pred_prob,        # [L, B]
            "layer_pred_label": layer_pred_label        # [L, B]
        }


    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, preds, target):
        sequences           = preds["sequences"]
        layer_pred_prob     = preds["layer_pred_prob"]   # [L, B]
        layer_pred_label    = preds["layer_pred_label"]  # [L, B]


        # ---------- per-layer *correct / wrong* bookkeeping ---------------
        #
        #   layer_correct:  True  → prediction matches ground-truth
        #                   False → wrong
        #
        #   Works for a *single* classification task (one target column).
        #   If you have multiple tasks, split `layer_pred_label`
        #   exactly the way you already do above when computing accuracy.
        #
        true_labels   = target[:, 0].long()                 # [B]
        layer_correct = (layer_pred_label == true_labels.unsqueeze(1))  # [L, B]

        results = {
            "layer_pred_prob": layer_pred_prob.detach().cpu(),   # tensor [L, B]
            "layer_correct"  : layer_correct.detach().cpu()      # bool   [L, B]
        }

        # optionally dump to CSV if you set an env-var
        record_fp = os.getenv("RESULT_CSV")
        print(f"record_fp: {record_fp}")
        if record_fp:
            import csv
            with open(record_fp, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_idx", "layer_idx", "probability", "correct"])
                B, L = layer_pred_prob.shape
                for s in range(B):
                    for l in range(L):
                        writer.writerow([
                            s,
                            l,
                            layer_pred_prob[s, l].item(),
                            int(layer_correct[s, l].item())
                        ])

        with open(os.getenv("result_file"), 'wb') as f:
            pickle.dump({"sequences": sequences, "layer_pred_prob": layer_pred_prob, "layer_pred_label": layer_pred_label}, f)

        return {results}


########################################
########################################
## Node


########################################!~!~!~!
# Normal

@R.register("tasks.ClassificationTemperature_Node_continuous")
class ClassificationTemperature_Node_continuous(tasks.Task, core.Configurable):
    _option_members = {"criterion", "metric"}

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("macro_auprc", "macro_auroc"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        verbose=0,
    ):
        super(ClassificationTemperature_Node_continuous, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, disable normalization
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose
        self.num_layers = 33

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute mean, std, and num_class on the training set,
        then build an MLP head per layer.
        """
        # Determine whether we are working at the node, atom, or residue level
        self.view = getattr(train_set[0]["graph"], "view", "atom")

        # Collect all target values from the train set for statistics
        values_list = []
        for data in train_set:
            values_list.append(data["graph"].target)  # shape: (num_nodes,) or (num_residues,)

        values = torch.cat(values_list, dim=0)
        mean = values.float().mean()
        std = values.float().std()

        # Figure out number of classes if doing classification
        num_class = 1
        if values.dtype == torch.long:
            # If max label is >1 or not using BCE, it means multiclass
            nmax = values.max().item()
            if nmax > 1 or "bce" not in self.criterion:
                nmax += 1
            num_class = nmax

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

    def predict(self, batch, all_loss=None, metric=None):
        """
        Return a list of predictions (one per layer).
        Each layer's node_feature is passed to a separate MLP.
        """

        graphs = batch["graph"]
        sequences = []
        device = next(self.model.parameters()).device
        n_layers  = 33

        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)

        B = len(sequences)

        layer = int(os.getenv("LAYER"))

        data_ = [(f"protein_{i}", s) for i, s in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(data_)
        batch_tokens = batch_tokens.to(device)
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)

        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.12
            src_len = (~padding_mask).sum(-1)
            mask_ratio_obs = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_len
            mask_ratio_obs = torch.clamp(mask_ratio_obs, min=1e-9)
            x *= ((1 - mask_ratio_train) / (1 - mask_ratio_obs)).unsqueeze(-1).unsqueeze(-1)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x)) 
        x = x.transpose(0, 1) 
        
        for lidx, layer_mod in enumerate(self.model.model.layers):
            h = layer_mod(x, self_attn_padding_mask = padding_mask, need_head_weights=False)[0]
            x = h
            if lidx == layer:
                break

        h_seq = h.transpose(0,1)
        logits_list = []
        for b in range(B):
            seq_len   = len(sequences[b])
            h_b       = h_seq[b, :seq_len, :]              # strip pad positions
            logits_b  = self.mlp[layer](h_b)              # (seq_len, num_classes)
            logits_list.append(logits_b)

        return {"pred": logits_list}
            
    def target(self, batch):
        """
        Return a dictionary with:
          "label": the node-level target
          "mask": a boolean mask indicating which nodes are labeled
          "size": used for some metrics requiring per-graph aggregates
        """
        graph = batch["graph"]
        size = graph.num_nodes if self.view in ["node", "atom"] else graph.num_residues
        return {
            "label": graph.target,   # shape: (num_nodes,) or (num_residues,)
            "mask": graph.mask,      # shape: (num_nodes,) or (num_residues,)
            "size": size
        }

    def evaluate(self, preds, target):
        """
        Evaluate each layer's predictions given the `target`.
        preds: list of tensors, each is shape (N, num_class) or (N,) depending on your MLP output
        target: dict with { "label", "mask", "size" }
        """
        metric = {}
        _target = target["label"]
        _mask = target["mask"]
        labeled = ~torch.isnan(_target) & _mask
        _size = functional.variadic_sum(labeled.long(), target["size"])
        pred = preds["pred"]

        for _metric in self.metric:
            if _metric in ["mae", "rmse"]:
                # Typically for regression
                if _metric == "mae":
                    score = F.l1_loss(pred, _target, reduction="none")
                else:  # rmse
                    score = F.mse_loss(pred, _target, reduction="none").sqrt()

                score = functional.masked_mean(score, labeled, dim=0)

            elif _metric in ["micro_auroc", "micro_auprc"]:
                # Single "micro" approach across all labeled nodes
                if _metric == "micro_auroc":
                    score = metrics.area_under_roc(pred[labeled], _target[labeled])
                else:
                    score = metrics.area_under_prc(pred[labeled], _target[labeled])

            elif _metric in ["macro_auroc", "macro_auprc"]:
                # "macro" means compute per-graph, then average
                if _metric == "macro_auroc":
                    score = metrics.variadic_area_under_roc(pred[labeled], _target[labeled], _size).mean()
                else:
                    score = metrics.variadic_area_under_prc(pred[labeled], _target[labeled], _size).mean()

            elif _metric == "macro_acc":
                # One typical approach for multi-class:
                # (pred[labeled].argmax(-1) == _target[labeled]).float()
                #print(f"labeled.shape {labeled.shape}")
                #print(f"target shape {_target.shape}")
                #pred = torch.cat(pred, dim=0)
                pred = torch.cat(pred, dim=0)
                #print(f"pred.shape {pred.shape}")
                pred_argmax = pred[labeled].argmax(dim=-1)
                correct = (pred_argmax == _target[labeled]).float()
                score = functional.variadic_mean(correct, _size).mean()
                metric["macro_acc"] = score.item()

            else:
                raise ValueError(f"Unknown metric `{_metric}`")

        return metric


########################################!~!~!~!
# Exit

@R.register("tasks.EarlyExitClassificationTemperature_Node_continuous")
class EarlyExitClassificationTemperature_Node_continuous(tasks.Task, core.Configurable):
    _option_members = {"criterion", "metric"}

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("macro_auprc", "macro_auroc"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        verbose=0,
    ):
        super(EarlyExitClassificationTemperature_Node_continuous, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, disable normalization
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose
        self.num_layers = 33

    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute mean, std, and num_class on the training set,
        then build an MLP head per layer.
        """
        # Determine whether we are working at the node, atom, or residue level
        self.view = getattr(train_set[0]["graph"], "view", "atom")

        # Collect all target values from the train set for statistics
        values_list = []
        for data in train_set:
            values_list.append(data["graph"].target)  # shape: (num_nodes,) or (num_residues,)

        values = torch.cat(values_list, dim=0)
        mean = values.float().mean()
        std = values.float().std()

        # Figure out number of classes if doing classification
        num_class = 1
        if values.dtype == torch.long:
            # If max label is >1 or not using BCE, it means multiclass
            nmax = values.max().item()
            if nmax > 1 or "bce" not in self.criterion:
                nmax += 1
            num_class = nmax

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

    def predict(self, batch, all_loss=None, metric=None):
        """
        Return a list of predictions (one per layer).
        Each layer's node_feature is passed to a separate MLP.
        """

        graphs = batch["graph"]
        sequences = []
        device = next(self.model.parameters()).device
        n_layers  = 33

        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)

        B = len(sequences)
        temp_file = os.getenv("TEMPERATURE_FILE")
        if temp_file and temp_file.lower() != "none":
            temps = torch.tensor(self.extract_temperatures(temp_file), device=device)              # shape (num_layers,)
        else:
            temps = torch.ones(self.model.model.num_layers, device=device)

        chosen_outputs = []
        chosen_layers = []
        threshold = float(os.getenv("CFG_THRESHOLD"))
        percent = float(os.getenv("PERCENT"))

        final_logits = [None] * B
        final_layers = torch.full((B,), -1, device=device) 
        best_logits = [None] * B
        best_prob = torch.full((B,), -float("inf"), device=device)
        best_layers = torch.full((B,), -1, device=device) 
        computed_layers = torch.full((B,), -1, device=device)

        data_ = [(f"protein_{i}", s) for i, s in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(data_)
        batch_tokens = batch_tokens.to(device)
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)

        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.12
            src_len = (~padding_mask).sum(-1)
            mask_ratio_obs = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_len
            mask_ratio_obs = torch.clamp(mask_ratio_obs, min=1e-9)
            x *= ((1 - mask_ratio_train) / (1 - mask_ratio_obs)).unsqueeze(-1).unsqueeze(-1)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x)) 
        x = x.transpose(0, 1) 

        threshold = float(os.getenv("CFG_THRESHOLD"))
        percent = float(os.getenv("PERCENT"))
        active = torch.arange(B, device=device)

        total = 0

        for lidx, layer in enumerate(self.model.model.layers):
            if active.numel() == 0:
                break
            for idx in active.tolist():
                computed_layers[idx] = lidx 
            h = x[:, active, :]
            h = layer(h, self_attn_padding_mask=padding_mask[active], need_head_weights=False)[0]
            x[:, active, :] = h
            h_seq = h.transpose(0,1)
            logits_list = []
            max_prob_for_mean = torch.empty(active.size(0), device=device)
            for local_idx, glob_idx in enumerate(active.tolist()):
                seq_len = len(sequences[glob_idx])
                h_i = h_seq[local_idx, :seq_len, :]
                logits_i = self.mlp[lidx](h_i)
                logits_list.append(logits_i)
                scaled = logits_i/temps[lidx]
                probs = torch.sigmoid(scaled)
                max_res = probs.max(dim=1).values
                max_prob_for_mean[local_idx] = max_res.mean()

                if (max_res > threshold).float().mean() >= percent:
                    final_logits[glob_idx] = logits_i
                    final_layers[glob_idx] = lidx

            done_mask = final_layers[active] != -1
            newly_done = active[done_mask]
            still_active = active[~done_mask]
            is_final = lidx == n_layers - 1

            if still_active.numel() > 0:
                better = max_prob_for_mean[~done_mask] > best_prob[still_active]
                if is_final and os.getenv("SELECT_LAST", "False") == "True":
                    better = torch.ones_like(better, dtype=torch.bool)

                if better.any():
                    upd_idx = still_active[better]
                    best_prob[upd_idx]   = max_prob_for_mean[~done_mask][better]
                    best_layers[upd_idx] = lidx
                    for k, g in enumerate(upd_idx.tolist()):
                        best_logits[g] = logits_list[(~done_mask).nonzero(as_tuple=True)[0][k]]
            active = still_active
        if active.numel() > 0:
            for g in active.tolist():
                final_logits[g] = best_logits[g]
                final_layers[g] = best_layers[g]

        ascii_mat = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor([ord(c) for c in s], device=device) for s in sequences],
                batch_first=True, padding_value=0,
            )
        if ascii_mat.size(1) < 2000:
            pad = ascii_mat.new_zeros(ascii_mat.size(0), 2000 - ascii_mat.size(1))
            ascii_mat = torch.cat([ascii_mat, pad], dim=1)
        return {
            "pred": final_logits,
            "layers": final_layers,
            "computed_layers": torch.tensor(computed_layers, device=self.device, dtype=torch.int64),
            "sequences": ascii_mat
        }
    
    def target(self, batch):
        """
        Return a dictionary with:
          "label": the node-level target
          "mask": a boolean mask indicating which nodes are labeled
          "size": used for some metrics requiring per-graph aggregates
        """
        graph = batch["graph"]
        size = graph.num_nodes if self.view in ["node", "atom"] else graph.num_residues
        return {
            "label": graph.target,   # shape: (num_nodes,) or (num_residues,)
            "mask": graph.mask,      # shape: (num_nodes,) or (num_residues,)
            "size": size
        }

    def evaluate(self, preds, target):
        """
        Evaluate each layer's predictions given the `target`.
        preds: list of tensors, each is shape (N, num_class) or (N,) depending on your MLP output
        target: dict with { "label", "mask", "size" }
        """
        metric = {}
        _target = target["label"]
        _mask = target["mask"]
        labeled = ~torch.isnan(_target) & _mask
        _size = functional.variadic_sum(labeled.long(), target["size"])
        pred = preds["pred"]

        layers = preds["layers"]
        layer_frequencies = torch.bincount(layers)
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total

        for _metric in self.metric:
            if _metric in ["mae", "rmse"]:
                # Typically for regression
                if _metric == "mae":
                    score = F.l1_loss(pred, _target, reduction="none")
                else:  # rmse
                    score = F.mse_loss(pred, _target, reduction="none").sqrt()

                score = functional.masked_mean(score, labeled, dim=0)

            elif _metric in ["micro_auroc", "micro_auprc"]:
                # Single "micro" approach across all labeled nodes
                if _metric == "micro_auroc":
                    score = metrics.area_under_roc(pred[labeled], _target[labeled])
                else:
                    score = metrics.area_under_prc(pred[labeled], _target[labeled])

            elif _metric in ["macro_auroc", "macro_auprc"]:
                # "macro" means compute per-graph, then average
                if _metric == "macro_auroc":
                    score = metrics.variadic_area_under_roc(pred[labeled], _target[labeled], _size).mean()
                else:
                    score = metrics.variadic_area_under_prc(pred[labeled], _target[labeled], _size).mean()

            elif _metric == "macro_acc":
                # One typical approach for multi-class:
                # (pred[labeled].argmax(-1) == _target[labeled]).float()
                #print(f"labeled.shape {labeled.shape}")
                #print(f"target shape {_target.shape}")
                #pred = torch.cat(pred, dim=0)
                pred = torch.cat(pred, dim=0)
                #print(f"pred.shape {pred.shape}")
                pred_argmax = pred[labeled].argmax(dim=-1)
                correct = (pred_argmax == _target[labeled]).float()
                score = functional.variadic_mean(correct, _size).mean()
                metric["macro_acc"] = score.item()

            else:
                raise ValueError(f"Unknown metric `{_metric}`")
            
            computed_layers = preds["computed_layers"]
            computed_layer_frequencies = torch.bincount(computed_layers)
            total_computed = computed_layer_frequencies.sum()
            computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
            average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed
            metric["avg_computed_layer"] = average_computed_layer.item()

            metric["layer"] = average_layer.item()

        return metric


########################################!~!~!~!
# Analysis














####END###




            


    








