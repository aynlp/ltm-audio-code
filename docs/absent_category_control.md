# Absent-Category Control

This optional analysis measures whether a selected category memory pulls a
model's free-form response toward that category when the category is absent
from the audio. It is a negative control for category-directional bias. It is
not a target-perception accuracy result and must not be merged into the main
classification or retrieval tables.

## Default design

The default configuration injects the `piano music` memory. It rejects a
mixture unless its target and all three interferers are outside the complete
music family: piano, flute, organ, classical guitar, violin, trumpet, clarinet,
alto saxophone, and drums. This matters because a response such as `Music`
cannot be attributed to an injected piano memory when a musical interferer is
present in the input.

The selected panel has 100 held-out mixtures. It uses 12 `alarm sound`
targets and 11 targets for each of `baby crying`, `car horn`, `cat meow`, `cow
moo`, `crow call`, `door slam`, `frog croak`, and `glass shatter`. The quota is
specified before decoding; it reflects which target categories have enough
four-source mixtures that satisfy the strict family-absence rule. Do not fill a
shortfall by relaxing the exclusion rule.

## Paired execution

For every selected mixture, using Qwen2-Audio:

1. Extract its audio tokens once at the model's pre-language-backbone
   interface.
2. Decode those unmodified tokens using the free-form prompt.
3. Load the model's calibrated injected-category memory, enhance the same
   tokens, and decode again with the same prompt and generation settings.

The runner fingerprints the configuration, manifest, memory archive, alias
mapping, adapter factory, and generation length. It refuses to append to a
mismatched response file. The optional `--verify-native-baseline` preflight is
available for adapters that can decode an untouched audio input natively; it
checks equivalence with the unmodified token-injection path before the paired
run.

## Reporting

The primary outcome is the proportion of answers uniquely normalized to the
injected category by the same pre-specified alias map used for free-form
classification. Report baseline and LTM rates, their paired difference,
transitions `n00`, `n01`, `n10`, `n11`, and the paired bootstrap 95% interval.

The raw word `piano` and a broader music-family expression are secondary
diagnostics. A rise in generic music wording alone is evidence of broad
semantic drift, not a piano-specific effect. An increase in injected-category
answers demonstrates a directional bias of the LTM condition; it does not
demonstrate improved perception.

## Release boundary

Only this protocol, configuration, source code, and adapter contract belong in
an anonymous release. Keep all manifests, audio, memory archives, model paths,
adapter argument files, response JSONL files, summaries, and logs local.
