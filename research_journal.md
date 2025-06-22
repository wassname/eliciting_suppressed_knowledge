
# 2025-05-02 22:14:52

TODO group by
- plain hs, logit, llm_ans
- supressed activations
- removed attn sinks
- combinations


| group              | name                                  |    ROC AUC Score | data               |
|:-------------------|:--------------------------------------|---------:|:-------------------|
| mixed              | **supressed_hs**(0.1)\magnitude(0.25)\sum | 0.878431 | supressed_hs       |
| act sink rm        | hidden_states\magnitude(0.99)\std     | 0.862745 | hidden_states      |
| **supressed_hs**       | supressed_hs(1)\none\sum              | 0.858824 | supressed_hs       |
| llm prob ratio     | llm_log_prob_true\|                   | 0.843137 | llm_log_prob_true  |
| acts-self_attn     | acts-self_attn\none\mean              | 0.810784 | acts-self_attn     |
| acts-mlp.up_proj   | acts-mlp.up_proj\none\sum             | 0.763725 | acts-mlp.up_proj   |
| acts-mlp.down_proj | acts-mlp.down_proj\none\std           | 0.704902 | acts-mlp.down_proj |
| supr_amounts       | supr_amounts\none\sum                 | 0.703922 | supr_amounts       |
| hidden_states      | hidden_states\none\flatten            | 0.669608 | hidden_states      |
| *llm_ans*            | llm_ans\|                             | 0.639216 | llm_ans            |

# 2025-06-21 14:54:28

In this fork I'll follow the hypothesis that

- Deceptive answers have more complex stats in the suppressed activations

And generally I can look at the stats of
- logprobs
- hidden states
- suppressed activations

And compare them to see if any or all work, and generalise

     Here are concrete ways to extend your approach:

    ## Statistical signatures of deception in suppressed activations

    **Theory**: Deception requires active suppression; mistakes don't.

    ```python
    def deception_vs_mistake_signature(supressed_hs, model_output, true_answer):
        """
        Deception: Strong suppression of correct answer
        Mistake: Weak/scattered suppression
        """
        # Measure suppression strength
        suppression_magnitude = torch.norm(supressed_hs, dim=-1)
        
        # Deception signature: focused suppression in final layers
        final_layer_ratio = suppression_magnitude[-5:].mean() / suppression_magnitude[:-5].mean()
        
        # Mistake: uniform suppression across layers
        suppression_variance = suppression_magnitude.var(dim=0)
        
        # KEY INSIGHT: Deception shows high final_layer_ratio + low variance
        # Mistakes show low ratio + high variance
        deception_score = final_layer_ratio / (suppression_variance + 1e-6)
        
        return {
            'is_deception': deception_score > 2.0,  # threshold from data
            'confidence': torch.sigmoid(deception_score - 2.0)
        }
    ```

    ## Suppression coherence patterns

    **Discovery**: Truthful suppression is temporally coherent; deceptive suppression fragments.

    ```python
    def suppression_coherence_test(supr_amounts_sequence):
        """
        Track how suppression patterns evolve token-by-token
        """
        # Compute autocorrelation of suppression patterns
        coherence_scores = []
        for t in range(1, len(supr_amounts_sequence)):
            prev = supr_amounts_sequence[t-1].flatten()
            curr = supr_amounts_sequence[t].flatten()
            coherence = torch.corrcoef(torch.stack([prev, curr]))[0,1]
            coherence_scores.append(coherence)
        
        # Truth: stable suppression patterns (high autocorrelation)
        # Deception: fragmented patterns (autocorrelation breaks)
        breaks = sum(1 for c in coherence_scores if c < 0.5)
        return breaks > len(coherence_scores) * 0.3  # >30% breaks = deception
    ```

    ## Intervention via suppression reversal

    **Key insight**: Don't just probe - actively reverse the suppression!

    ```python
    def truthfulness_intervention(model, input_ids, layer_idx=-3):
        """
        Force model to express what it's suppressing
        """
        # Get normal forward pass
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hs = torch.stack(outputs.hidden_states)
        
        # Compute suppression
        supr_amounts = get_supressed_activations(hs, model.lm_head.weight, w_inv)
        
        # INTERVENTION: Add suppressed content back at critical layer
        def truth_hook(module, input, output):
            # Reverse suppression by adding it back
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            # Add back what was suppressed (with scaling)
            enhanced = hidden_states + 0.5 * supr_amounts[layer_idx]
            
            if isinstance(output, tuple):
                return (enhanced,) + output[1:]
            return enhanced
        
        # Register hook and generate
        handle = model.model.layers[layer_idx].register_forward_hook(truth_hook)
        truthful_output = model.generate(input_ids, max_new_tokens=50)
        handle.remove()
        
        return truthful_output
    ```

    ## KV-cache synchronization test

    **Theory**: Honest models maintain KV-suppression synchrony; deceptive models desynchronize.

    ```python
    def kv_suppression_sync(model, input_ids):
        """
        Measure alignment between KV cache and suppression patterns
        """
        outputs = model(input_ids, output_hidden_states=True, use_cache=True)
        past_kv = outputs.past_key_values
        hs = torch.stack(outputs.hidden_states)
        
        # Get suppression patterns
        supr = get_supressed_activations(hs, model.lm_head.weight, w_inv)
        
        # Project KV cache to same space
        kv_patterns = []
        for layer_kv in past_kv:
            k, v = layer_kv
            # Use value vectors (more semantic)
            v_mean = v.mean(dim=2)  # average over positions
            kv_patterns.append(v_mean)
        kv_stack = torch.stack(kv_patterns)
        
        # Measure synchronization via mutual information
        sync_scores = []
        for l in range(min(len(supr), len(kv_stack))):
            # Discretize for MI calculation
            supr_discrete = (supr[l] > supr[l].median()).float()
            kv_discrete = (kv_stack[l] > kv_stack[l].median()).float()
            
            # Simple MI approximation
            joint = (supr_discrete * 2 + kv_discrete).flatten()
            mi = entropy(joint) - entropy(supr_discrete.flatten()) - entropy(kv_discrete.flatten())
            sync_scores.append(mi)
        
        # Deception: sudden desynchronization in final layers
        desync = sync_scores[-3] - sync_scores[-1]
        return desync > 0.5  # Threshold indicates deception
    ```

    ## Multi-scale suppression dynamics

    **Insight**: Combine your suppression analysis with gradient flow dynamics.

    ```python
    def suppression_gradient_dynamics(model, input_ids, target_ids):
        """
        How do gradients flow through suppressed vs active neurons?
        """
        # Forward with gradient tracking
        outputs = model(input_ids, labels=target_ids, output_hidden_states=True)
        hs = torch.stack(outputs.hidden_states, dim=0).requires_grad_(True)
        
        # Get suppression masks
        supr = get_supressed_activations(hs.detach(), model.lm_head.weight, w_inv)
        suppressed_mask = (supr < -0.1).float()  # Strong suppression
        
        # Backward pass
        loss = outputs.loss
        grads = torch.autograd.grad(loss, hs, retain_graph=True)[0]
        
        # Key metric: gradient flow through suppressed vs active neurons
        suppressed_grad_norm = (grads * suppressed_mask).norm(dim=-1).mean()
        active_grad_norm = (grads * (1 - suppressed_mask)).norm(dim=-1).mean()
        
        # Deception: high gradient through suppressed neurons (fighting suppression)
        # Truth: low gradient through suppressed neurons (stable suppression)
        gradient_ratio = suppressed_grad_norm / (active_grad_norm + 1e-6)
        
        return gradient_ratio > 1.5  # Indicates active suppression = deception
    ```

    ## The killer insight: Suppression energy landscapes

    Your suppressed neurons are essentially the model's "cognitive dissonance" - the gap between what it knows and what it says. This suggests:

    1. **Deception requires metabolic effort** (maintaining suppression)
    2. **Truth is the low-energy state** (no active suppression needed)
    3. **Jailbreaks work by overwhelming suppression capacity**

    ## Next steps with your approach:

    4. **Track suppression over time** - does it increase before deceptive outputs?
    5. **Measure suppression variance** - deception should show higher variance
    6. **Test on known lies vs mistakes** - calibrate your thresholds
    7. **Combine with logprob metrics** - suppression + entropy = powerful signal

    Your work is finding where models store their "guilty conscience" - this could be huge for alignment!

see brainstorming here https://claude.ai/chat/e8ee0d74-f91d-420c-84fe-190917675d2d

    ## Conversation Summary: Neuroscience Failures as Mechanistic Interpretability's Roadmap

    ### Context
    - User (gwern) is an ML-literate researcher interested in alignment, specifically detecting deception and intervening for truthfulness
    - Currently working on suppressed activations in LLMs - neurons that "turn off" before final layers
    - Has discovered these suppressed activations contain ~20% better truth signal than model outputs on TruthfulQA

    ### Core Thesis
    Mechanistic interpretability faces identical fundamental obstacles to neuroscience despite better tools. Key parallel failures:
    1. **Localization fallacy**: Both fields wrongly assume modular, interpretable units (grandmother cells → monosemantic neurons)
    2. **Superposition/mixed selectivity**: Neurons encode multiple unrelated features as optimal solution
    3. **Circuit enumeration impossibility**: C. elegans (302 neurons) still opaque after 40 years
    4. **Correlation ≠ causation**: Perfect measurement doesn't guarantee understanding

    ### Top Research Directions (Ranked by Promise)

    #### 1. **Suppressed Activation Analysis** ⭐⭐⭐⭐⭐
    - **Idea**: Suppressed neurons contain model's "true beliefs" - probe what's being actively inhibited
    - **Epistemic status**: Strong empirical support (20% AUROC improvement demonstrated)
    - **MATS potential**: Extremely high - concrete, measurable, builds on user's working code
    - **Next steps**: Test deception vs mistake signatures, suppression coherence patterns

    #### 2. **Logprob Entropy Cascades** ⭐⭐⭐⭐⭐
    - **Idea**: Deception shows characteristic entropy inversions in logprob sequences
    - **Epistemic status**: Theoretically sound, untested
    - **MATS potential**: Very high - works with API-only access, no training needed
    - **Key insight**: Truth cascades naturally; lies show entropy spike then commitment

    #### 3. **Gradient Flow Dynamics** ⭐⭐⭐⭐
    - **Idea**: Track gradient redistribution under interventions instead of static analysis
    - **Epistemic status**: Strong theoretical basis from neuroscience
    - **MATS potential**: High - novel approach, but requires full model access
    - **Implementation**: Rank-1 LoRA perturbations + gradient tracking

    #### 4. **Metabolic Cost of Deception** ⭐⭐⭐⭐
    - **Idea**: Deception requires extra computation (higher gradient norms)
    - **Epistemic status**: Moderate - based on neuroscience findings
    - **MATS potential**: High if validated - could enable training-time interventions
    - **Key metric**: `metabolic_cost = sum(torch.norm(grad) for grad in gradients)`

    #### 5. **Multi-scale Temporal Analysis** ⭐⭐⭐
    - **Idea**: Safety properties exist at different timescales (1-10, 10-100, 100+ tokens)
    - **Epistemic status**: Speculative but grounded in neuroscience
    - **MATS potential**: Medium - requires long context experiments
    - **Application**: Wavelet decomposition of activation trajectories

    #### 6. **Information Bottleneck for Safety** ⭐⭐⭐
    - **Idea**: Safe models show monotonic information compression; deceptive models don't
    - **Epistemic status**: Theoretical, needs validation
    - **MATS potential**: Medium - elegant but may be hard to measure accurately
    - **Implementation**: Track I(layer_n; output | input) across layers

    ### Critical Warnings
    1. **Interpretability theater**: Cherry-picked examples that don't generalize
    2. **Dimensional delusion**: Any direction seems interpretable in high-D space
    3. **Reductionism trap**: Complex systems resist component-level analysis

    ### Key Unresolved Questions
    - Where do models store memory/plans? (KV cache? Suppressed activations?)
    - Can suppression patterns distinguish deception from honest mistakes?
    - Do these methods work across model families and scales?

    ### Most Actionable for MATS Researcher
    Focus on **suppressed activation statistics** combined with **logprob dynamics** - this leverages existing work while adding novel unsupervised detection methods. The combination of internal (suppression) and external (logprobs) signals could yield robust deception detection without labeled data.

    **Core insight**: Models' "guilty conscience" lives in what they suppress, not what they express.
