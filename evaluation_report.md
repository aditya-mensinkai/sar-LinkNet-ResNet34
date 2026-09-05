# Sentinel-1 validation evaluation

Best checkpoint epoch: 31

- IoU/Jaccard: 0.810690
- Dice: 0.895448
- Precision: 0.886202
- Recall: 0.904890
- F1: 0.895448

`predictions_sample.png` contains qualitative validation examples. Mumbai inference is qualitative only; no Mumbai ground truth or accuracy claim is included.

## Qualitative observations

The displayed validation examples show good recovery of broad spill-shaped regions. Thin or very small targets can fragment or disappear, and some larger predicted regions have rough boundaries or interior holes. These observations are qualitative only and should be rechecked on additional held-out tiles before using the model operationally.
