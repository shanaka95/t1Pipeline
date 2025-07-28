# Extended Clustering Analysis Summary (5-199 Clusters)

## 🎯 Key Findings

### Best Performing Cluster Numbers
1. **Cluster 121**: 0.665258 (Best overall)
2. **Cluster 112**: 0.655963
3. **Cluster 155**: 0.655922
4. **Cluster 107**: 0.652870
5. **Cluster 110**: 0.652566

### 📊 Analysis Statistics
- **Total clusters analyzed**: 195 (from 5 to 199)
- **Best Silhouette Score**: 0.665258 (Cluster 121)
- **Worst Silhouette Score**: 0.325082 (Cluster 5)
- **Average Silhouette Score**: 0.584145
- **Median Silhouette Score**: 0.603723
- **Standard Deviation**: 0.063415

## 📈 Performance Trends

### By Cluster Ranges
- **5-20 clusters**: Poor performance (avg: ~0.45)
- **21-50 clusters**: Improving performance (avg: ~0.54)
- **51-100 clusters**: Good performance (avg: ~0.61)
- **101-150 clusters**: **Best performance** (avg: ~0.63)
- **151-199 clusters**: Slightly declining (avg: ~0.60)

### Key Insights
1. **Optimal Range**: 100-150 clusters shows the best performance
2. **Performance Plateau**: Scores plateau around 100-150 clusters
3. **Diminishing Returns**: Beyond 150 clusters, performance starts to decline
4. **Sweet Spot**: Cluster numbers around 110-130 perform exceptionally well

## 🏆 Top 10 Performers

| Rank | Cluster | Silhouette Score |
|------|---------|------------------|
| 1    | 121     | 0.665258         |
| 2    | 112     | 0.655963         |
| 3    | 155     | 0.655922         |
| 4    | 107     | 0.652870         |
| 5    | 110     | 0.652566         |
| 6    | 136     | 0.651274         |
| 7    | 183     | 0.651259         |
| 8    | 134     | 0.647211         |
| 9    | 99      | 0.646775         |
| 10   | 111     | 0.646743         |

## 📊 Visualizations Created

1. **extended_silhouette_vs_clusters.png**: Main scatter plot showing all scores vs cluster numbers
2. **performance_by_ranges.png**: Bar chart comparing performance across cluster ranges
3. **top_20_performers.png**: Detailed view of the top 20 performing cluster numbers
4. **score_distribution.png**: Histogram showing the distribution of all scores
5. **performance_trend_analysis.png**: Trend analysis with moving averages and improvements

## 🎯 Recommendations

### Primary Recommendation
**Use Cluster 121** for optimal performance with a Silhouette Score of 0.665258

### Alternative Options
- **Cluster 112**: 0.655963 (Excellent performance)
- **Cluster 155**: 0.655922 (Excellent performance)
- **Cluster 107**: 0.652870 (Very good performance)
- **Cluster 110**: 0.652566 (Very good performance)

### Avoid
- Cluster numbers below 50 (poor performance)
- Very high cluster numbers (>150) show diminishing returns

## 🔍 Comparison with Previous Analysis

### Previous Best (5-100 clusters)
- **Best**: Cluster 99 with 0.646775

### Extended Analysis (5-199 clusters)
- **Best**: Cluster 121 with 0.665258
- **Improvement**: +0.018483 (2.86% improvement)

## 📁 Files Generated

### Analysis Files
- `extended_clustering_analysis.json`: Raw analysis data
- `extended_clustering_analysis_report.txt`: Detailed analysis report
- `extended_clustering_summary.md`: This summary file

### Visualizations
- `extended_clustering_visualizations/`: Directory containing all PNG visualizations

## 🚀 Next Steps

1. **Use Cluster 121** for your main clustering analysis
2. **Test visualization** with the optimal cluster number
3. **Consider computational trade-offs** when choosing between top performers
4. **Validate results** with domain-specific evaluation metrics

## 💡 Key Takeaways

1. **Extended range reveals better options**: Going beyond 100 clusters found significantly better performers
2. **Sweet spot identified**: 110-130 clusters show consistently excellent performance
3. **Performance plateau**: Beyond 150 clusters, improvements become marginal
4. **Significant improvement**: Best score improved by 2.86% compared to previous analysis
5. **Robust methodology**: The top 10 performers all score above 0.64, indicating reliable clustering

---

*Analysis completed on: July 28, 2024*
*Total clusters analyzed: 195 (5-199)*
*Best performing cluster: 121 (Silhouette Score: 0.665258)* 