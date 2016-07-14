# FTIM - Feature-Time Instability Metric

FTIM is a method to test a feature's instability (i.e. how much the properties of the feature change over time).

It does this by segmenting the data based on time, and then looks at how much the histogram of positive/negative classes changes over the course of these partitions (when set to histogram intersection) or by looking at how much the purity of a split would change if a tree algorithm was to make a split in the data there.

This was very useful in the [Avito Duplicate Ads Detection](https://www.kaggle.com/c/avito-duplicate-ads-detection) competition for quickly checking which of our features might overfit to the leaderboard - and provided almost identical results to running XGBoost with a validation set on each feature individually, with over 1000x the speed.

## Options

`method` can either be set to 'inter' or 'purity'

Inter (histogram intersection) compares two time splits by further splitting them based on target value, and then computing histograms. For both positive and negative class histograms, the histogram intersection is computed between the time splits, and then ftim is returned as 1-mean of the histogram intersections.

Purity (split purity) compares two time splits by computing histograms, and then computing a probability for the target in each histogram bin. (This is comparable to how a Random Forest creates splits). For each bin, the change in probability is measured over time and then the changes are weighted by how many samples exist in the bin (if this option is set). ftim is returned as the total weighted difference in probability.

`time_res` is the resolution of the time segmentation. FTIM will split the data this many times sequentially, and compute stability metrics between each of these splits. Higher values are more sensitive to small fluctuations in time, but may give more noisy results.

`x_res` is the resolution of the feature splitting. This is the number of bins that the feature values will be placed into before computing the instability. Higher values are more sensitive and will find smaller shifts in values, but may give more noisy results.

`thresh` is the proportion of samples that a histogram bin must have to be considered in the calculations. This option is only used by the 'inter' method.

`weighted` decides whether the histogram bins are weighted by the number of samples that are in them when doing split purity calculations. This option is only used by the 'purity' method and is highly recommended (otherwise a bin which has 1 sample -> 0 samples is penalised the same as a bin which has 1000 samples -> 0 samples)

`ignore_zero` decides whether a bin is ignored completely when it has zero samples. Only used by the 'purity' method. I got some wonky results when turning this on - so it is probably better to keep this off and turn on weighting instead.
