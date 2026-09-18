| model | accuracy | precision | recall | specificity | mcc | auc_roc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.9757 | 0.9812 | 0.9874 | 0.9362 | 0.9307 | 0.9912 |
| deep_conv_lstm | **0.9806** | 0.9844 | **0.9906** | 0.9468 | **0.9445** | 0.9883 |
| wide_bilstm | 0.9709 | **0.9873** | 0.9748 | **0.9574** | 0.9189 | 0.9888 |
| extra_variant_1 | 0.9757 | 0.9843 | 0.9843 | 0.9468 | 0.9311 | **0.9916** |
| extra_variant_2 | 0.9757 | 0.9843 | 0.9843 | 0.9468 | 0.9311 | 0.9882 |

### Best model by metric
- **accuracy**: deep_conv_lstm
- **precision**: wide_bilstm
- **recall**: deep_conv_lstm
- **specificity**: wide_bilstm
- **mcc**: deep_conv_lstm
- **auc_roc**: extra_variant_1
