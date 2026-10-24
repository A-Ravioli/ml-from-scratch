# 27 · checkpoint: large language models

pass this to finish the chapter, or to skip it if you already know the material. no notes, no code from the chapter, no internet. a pen, paper and a calculator are fine for parts 1 and 2.

## 1. concept check

answer each in two or three sentences, then check.

1. why does the agreement rate of two MinHash signatures estimate the Jaccard similarity of the underlying sets?

<details><summary>answer</summary>

for one random hash, the minimum over $A \cup B$ is equally likely to be any element; the two minima agree exactly when that element is in $A \cap B$, which happens with probability $|A \cap B| / |A \cup B|$. averaging over $n$ independent hashes estimates it with standard deviation $\sqrt{J(1-J)/n}$.

</details>

2. what does LSH banding buy over comparing every pair of signatures, and why does a deduplication pipeline still verify candidate pairs?

<details><summary>answer</summary>

only documents sharing a bucket in some band are compared, so the work is about linear in the number of documents instead of quadratic. the S-curve $1 - (1 - s^r)^b$ lets some dissimilar pairs collide (32 bands of 4 make a 0.5-similar pair a candidate 87% of the time); verifying with the full signature discards them.

</details>

3. according to Muennighoff et al., what are 4 epochs over $U$ unique tokens worth, and what's the most any amount of repetition can be worth?

<details><summary>answer</summary>

$U + U R^*(1 - e^{-3/R^*}) \approx 3.7U$ with $R^* \approx 15.4$: almost as good as fresh data. the limit is $U(1 + R^*) \approx 16U$.

</details>

4. why is SwiGLU's hidden width $8d/3$, and why does a Llama have no position embedding?

<details><summary>answer</summary>

three matrices of $d \times d_\text{ff}$ hold as many parameters as a two-matrix $4d$ MLP when $d_\text{ff} = 8d/3$. rotary embeddings inside every attention layer rotate queries and keys by position, so scores depend on relative position; nothing needs to be added to the input.

</details>

5. where in grouped-query attention do you append to the KV cache, and why does it matter?

<details><summary>answer</summary>

before `repeat_kv`, storing the $H_{kv}$ distinct key/value heads. appending after the repetition stores identical copies and gives back the memory saving that is GQA's whole point.

</details>

6. why does an MoE router need an auxiliary loss, and why does that loss use the router's probabilities and not just the token counts?

<details><summary>answer</summary>

routing reinforces itself (experts learn from the tokens they get, so popular experts get better and more popular), and without a penalty a few experts take most tokens. counts come from a top-$k$ selection with no gradient; the mean router probabilities $P_e$ are differentiable, so $E\sum_e f_e P_e$ pushes probability away from overloaded experts.

</details>

7. in LoRA, why is $B$ initialized to zero and $A$ to random values?

<details><summary>answer</summary>

$B = 0$ makes the wrapped layer start exactly at the pretrained function. $A$ random makes the gradient with respect to $B$ (proportional to $(Ax)^\top$) nonzero, so training can start; with both zero neither would ever move.

</details>

8. in an SFT example, which targets are $-100$, and what goes wrong if the end-of-text token isn't a target?

<details><summary>answer</summary>

every target that is a prompt token (all but the last prompt position's target), so the loss is on the response only. without the end-of-text target, the model never learns when to stop and runs on into new turns.

</details>

9. state the DPO loss and say what plays the role of the reward.

<details><summary>answer</summary>

$-\log\sigma\big(\beta[\log\frac{\pi(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi(y_l|x)}{\pi_\text{ref}(y_l|x)}]\big)$, averaged over pairs. the implicit reward is $\beta\log\frac{\pi(y|x)}{\pi_\text{ref}(y|x)}$, from the closed-form optimum of the KL-regularized RLHF objective.

</details>

10. why does a policy optimized hard against a learned reward model eventually get worse at what you actually wanted?

<details><summary>answer</summary>

the reward model is a proxy fitted to answers near the starting policy. optimization moves the policy to where the proxy's errors are largest and in its favour (outside its training distribution), so the proxy keeps rising while the true objective peaks and falls (Gao et al., 2022). a KL penalty, early stopping on an independent measure, and better reward models delay it.

</details>

11. why is the output of speculative decoding an exact sample from the target model, whatever the draft?

<details><summary>answer</summary>

a drafted token $x \sim q$ is kept with probability $\min(1, p(x)/q(x))$ and otherwise replaced by a sample from $\max(0, p - q)$ renormalized; the emitted token's probability is $\min(p, q) + \max(0, p - q) = p$. the draft only changes how many tokens each target pass yields.

</details>

12. why isn't constrained decoding the same as sampling from the model conditioned on valid output?

<details><summary>answer</summary>

the mask renormalizes each step's distribution over tokens that can still lead to valid output, without weighting them by how likely the model is to complete them validly. prefixes the model likes for other reasons absorb probability that conditioning would have given to other valid strings.

</details>

13. why should two models evaluated on the same items be compared with a paired bootstrap?

<details><summary>answer</summary>

their per-item correctness is correlated through item difficulty. resampling the same items for both keeps the correlation, so the interval for the difference reflects only the items where they disagree; independent resampling double-counts the difficulty variance and gives far too wide an interval.

</details>

14. why can't any reader in a RAG system beat the retriever's recall@k, and how do agent loops help with multi-hop questions?

<details><summary>answer</summary>

without the evidence in its context the reader can only guess or rely on memory, so accuracy $\le$ recall (on unfamiliar facts). a multi-hop question's later evidence isn't findable from the question alone; an agent retrieves, reads the intermediate answer, and searches again with it, at the cost of multiplying each hop's failure rate.

</details>

## 2. by hand

1. **LSH.** with 128 hashes split into 16 bands of 8 rows, compute the probability that a pair with Jaccard 0.9 becomes a candidate, and the same for 0.5. roughly where does the curve cross one half?

<details><summary>solution</summary>

$0.9^8 = 0.430$, $(1 - 0.430)^{16} = 0.570^{16} \approx 1.2 \times 10^{-4}$, so $P \approx 0.9999$. $0.5^8 = 0.0039$, $(0.9961)^{16} = 0.939$, so $P = 0.061$. the threshold estimate $(1/16)^{1/8} = 0.71$ is where $P \approx 0.63$; solving $1 - (1 - s^8)^{16} = 1/2$ gives $s^8 = 1 - 2^{-1/16} = 0.042$, $s = 0.67$.

</details>

2. **LoRA and GQA counts.** a model has 32 layers, width 4096, 32 query heads of width 128 and 8 key/value heads, SwiGLU width 11,008. (a) how many parameters does a rank-16 LoRA on $q$, $k$, $v$, $o$ of every layer add? (b) how many bytes does the KV cache take per token in 16-bit, and how much would MHA (32 key/value heads) take?

<details><summary>solution</summary>

(a) $q$, $o$: $16(4096 + 4096) = 131{,}072$ each; $k$, $v$: $16(4096 + 1024) = 81{,}920$ each; per layer 425,984; times 32: 13.6M parameters. (b) $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072$ bytes (128 KiB) per token; MHA: 4× that, 512 KiB.

</details>

3. **DPO and speculative decoding.** (a) $\beta = 0.2$; the policy gives the chosen and rejected responses log-probabilities $-20$ and $-25$, the reference $-22$ and $-23$. compute the loss and the implicit rewards. (b) a draft is accepted with probability 0.7 per token; with $k = 3$ drafts, how many tokens does each target call produce on average?

<details><summary>solution</summary>

(a) implicit rewards $0.2(-20 + 22) = 0.4$ and $0.2(-25 + 23) = -0.4$; margin 0.8; loss $\log(1 + e^{-0.8}) = 0.371$. (b) $(1 - 0.7^4)/(1 - 0.7) = (1 - 0.2401)/0.3 = 2.53$.

</details>

## 3. blind rebuild

in 60 minutes, with no notes, rebuild in a fresh copy of the topic notebooks:

- `LoRALinear`, `apply_lora` and `merge_lora` (topic 03, stages 1–3), and
- `sequence_logprobs` and `dpo_loss` (topic 04, stages 1 and 3).

check them with the topics' own tests:

```bash
pytest 27-large-language-models/03-fine-tuning/test_exercise.py -k "stage1 or stage2 or stage3"
pytest 27-large-language-models/04-preference-learning/test_exercise.py -k "stage1 or stage3"
```

## 4. readiness

move on when you score at least 80% on parts 1 and 2 and the blind rebuild passes every check. the same checkpoint works as a placement test: if you pass it cold, skim the labs (each has at least one result worth seeing: duplicated data that lowers training loss and raises held-out loss, a learning rate that explains most of an "architecture" gap, LoRA that matches full fine-tuning on one task and fails on another, a policy that discovers "thee thee thee", speculative decoding that loses to a KV cache, and constrained decoding that isn't conditioning) and go on to chapter 28.
