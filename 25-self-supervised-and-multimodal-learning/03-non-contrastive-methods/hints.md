# non-contrastive methods · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`F.cosine_similarity(a, b, dim=1)` gives one cosine per row. SimSiam: `-(cos(p1, z2.detach()).mean() + cos(p2, z1.detach()).mean()) / 2`. BYOL: `(2 - 2 * cos(p, z.detach())).mean()`.

</details>

## stage 2

<details><summary>hint</summary>

work on `z.detach().cpu().double()` inside `torch.no_grad()`. `std = F.normalize(z, dim=1).std(dim=0, correction=0).mean().item()`. `s = torch.linalg.svdvals(z)`, `p = s / s.sum() + 1e-12`, `erank = torch.exp(-(p * p.log()).sum()).item()`.

</details>

## stage 3

<details><summary>hint</summary>

compute `z1, z2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))` and `p1, p2 = self.predictor(z1), self.predictor(z2)`. with the stop-gradient call `simsiam_loss(p1, p2, z1, z2)`; without it, write the same formula with `F.cosine_similarity` and no `.detach()`. return `(loss, z1)`.

</details>

## stage 4

<details><summary>hint</summary>

build the online projector first, then `self.target_encoder = copy.deepcopy(encoder)` and `self.target_projector = copy.deepcopy(self.projector)`, and loop over their parameters with `p.requires_grad_(False)`. in `forward`, compute the targets inside `with torch.no_grad():`. `update_target` is two calls: `ema_update(self.target_encoder, self.encoder, decay=self.momentum)` and the same for the projectors.

</details>

## stage 5

<details><summary>hint</summary>

it's topic 02's `train_simclr` loop with `loss, z1 = model(augment(xb), augment(xb))` in place of the head and InfoNCE, `Adam([p for p in model.parameters() if p.requires_grad], lr=lr)`, and after `opt.step()`: `if hasattr(model, "update_target"): model.update_target()`. append `loss.item()` and `collapse_metrics(z1)["std"]` to the history each step.

</details>
