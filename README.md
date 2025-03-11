# Eliciting Suppressed Knowledge (ESK)

## Abstract
Transformer models possess knowledge they actively suppress during inference. By isolating and probing these suppressed activation patterns, we demonstrate ~10% improvements on TruthfulQA compared to standard activation probes. This confirms LLMs encode truthful information in representations that are deliberately suppressed during generation, suggesting new approaches for mechanistic interpretability and truth extraction.

## Background
Recent mechanistic interpretability research identifies competing neural dynamics in transformers:

- Suppression neurons decrease probabilities of related tokens, counterbalancing prediction neurons that promote specific continuations (Gurnee et al., [2024](https://arxiv.org/pdf/2401.12181))

- This suppression-prediction dynamic creates an internal "debate" where certain activation patterns are consistently inhibited (Lad et al., [2024](https://arxiv.org/html/2406.19384))

- The architecture systematically develops prediction neurons until the final layers, which show "a sudden shift towards a much larger number of suppression neurons" (Gurnee et al., [2024](https://arxiv.org/pdf/2401.12181))

Previous work focused on identifying dedicated suppression neurons; we instead examine what information is being transiently suppressed during specific inferences—revealing knowledge deliberately inhibited during standard generation.

## Hypothesis
Transformer models maintain dual representations of knowledge: the dominant pathway that produces outputs, and suppressed activation patterns that encode alternative (often more truthful) representations. By directly probing these suppressed activations, we can extract knowledge the model "knows" but actively chooses not to express—much like extracting a dissenting opinion forcibly silenced during internal deliberation.

## Method
Our approach isolates suppressed activations by leveraging the differential impact of layers on token probabilities:

1. **Hidden State Extraction**: We collect layer-wise hidden states from the model.

2. **Logit Effect Isolation**: Following Gurnee et al.'s approach for identifying suppression neurons, we examine how each layer's contribution affects token probabilities by projecting hidden states through the output embedding matrix.

3. **Negative Differential Identification**: We compute layer-to-layer differences in logit space, identifying negative changes that represent suppressed information.

4. **Activation Space Projection**: Using the pseudo-inverse of the output embedding matrix, we project these negative changes back to activation space, revealing which hidden dimensions are being actively suppressed.

5. **Suppression-Masked Representation**: We apply this suppression mask to the original hidden states, isolating activation patterns that would increase truthful responses but are being down-weighted in the final layers.

6. **Linear Probing**: We train linear probes on these suppressed activation patterns and compare against standard probes and direct model outputs.

This method directly operationalizes the "residual sharpening" stage identified by Lad et al., exploiting the final layers' suppression dynamics to recover information that exists in the model but is being actively filtered out.

## Key Results

![TruthfulQA Performance Comparison](figures/truthfulqa_performance.png)

Linear probes targeting suppressed activations consistently outperform both naive outputs and standard activation probes across model scales. The performance gap (~10%) represents recoverable truthful knowledge that remains encoded but deliberately suppressed during normal generation.

### Performance Breakdown
| Method | ROC AUC Score |
|--------|---------------|
| LLM direct output | 0.53 |
| Hidden states (regular probe) | 0.62 |
| **Suppressed activations (ESK)** | **0.64** |

The highest-performing probe was on the final layer's suppressed activations (`hs_sup last`), supporting the hypothesis that the final "residual sharpening" stage specifically suppresses certain information pathways.

## Theoretical Context

Our findings provide empirical support for several key hypotheses about transformer mechanisms:

1. **Ensemble Hypothesis**: Both papers suggest that prediction and suppression neurons form competing ensembles. Our results empirically demonstrate that these competing pathways encode different information, with suppressed pathways often containing more truthful representations.

2. **Final Layer Calibration**: The "residual sharpening" phase appears to function not just as a confidence calibrator, but potentially as an alignment mechanism that modulates factuality in favor of other objectives. The performance improvement from probing suppressed activations quantifies this effect.

3. **Dual Representation**: Transformers maintain parallel representations of knowledge - one that is expressed in outputs and another that is encoded but suppressed. This suggests models contain more truthful knowledge than they express, contradicting simpler hypotheses that attribute factual errors purely to knowledge limitations.

4. **Stage-Specific Information**: The staged inference hypothesis proposed by Lad et al. predicts that different types of information would be emphasized at different model depths. Our results confirm this by showing that truth-related information is present but specifically suppressed during the final "residual sharpening" stage.


## Future Research Directions

This work opens several promising research avenues:

1. **Architectural Variations**: Investigate whether models with different suppression neuron densities show different patterns of truth suppression.

2. **Intervention Techniques**: Develop methods to selectively attenuate suppression dynamics during inference to improve factuality without full retraining.

3. **Suppression Categories**: Investigate whether specific categories of information (e.g., controversial facts, specialized knowledge) are more consistently suppressed than others.

4. **Layer Deletion Effects**: Explore how surgical modifications to final-layer suppression mechanisms affect factuality.

5. **Cross-Model Universality**: Determine whether suppressed activation patterns are universal across model architectures.

6. **RLHF Impact Analysis**: Investigate whether RLHF fine-tuning primarily operates by enhancing suppression mechanisms rather than modifying core knowledge.

7. **Content-Specific Suppression**: Analyze whether politically or ethically sensitive topics show stronger suppression effects than neutral factual questions.

## Citation

```
@software{clark2025esk,
  author = {Michael J Clark},
  title = {Eliciting Suppressed Knowledge: Recovering Truth from Suppressed Activations in Language Models},
  url = {https://github.com/wassname/eliciting_suppressed_knowledge},
  year = {2025},
}
```

## License
[MIT License](LICENSE)
