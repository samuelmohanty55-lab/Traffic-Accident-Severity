# Key Findings: Conditions Linked to Serious Road Traffic Accidents

This report details the statistical risk factors and machine learning explainability findings (Phase 8).

### 1. Statistical Odds Ratios & Relative Impact

| Risk Factor                     |   Odds Ratio (OR) |   95% CI Lower |   95% CI Upper |   Rate Exposed (%) |   Rate Baseline (%) |
|:--------------------------------|------------------:|---------------:|---------------:|-------------------:|--------------------:|
| Low Visibility (< 2 mi)         |              4.02 |           3.47 |           4.65 |               53.5 |                22.3 |
| Nighttime Driving (8 PM - 6 AM) |              1.9  |           1.8  |           2    |               29.8 |                18.3 |
| Highway Junction                |              1.9  |           1.77 |           2.03 |               33.9 |                21.3 |
| Rush Hour Commute               |              0.69 |           0.64 |           0.74 |               18   |                24.2 |
| Weekend Travel                  |              0.97 |           0.91 |           1.03 |               22.7 |                23.2 |
| Traffic Signal Present          |              0.66 |           0.61 |           0.71 |               17.4 |                24.3 |

### 2. Plain-Language Practical Takeaways

- Accidents with 'Low Visibility (< 2 mi)' have 4.02x higher odds of being serious (53.5% vs 22.3%).
- Accidents with 'Nighttime Driving (8 PM - 6 AM)' have 1.90x higher odds of being serious (29.8% vs 18.3%).
- Accidents with 'Highway Junction' have 1.90x higher odds of being serious (33.9% vs 21.3%).
- Presence of 'Rush Hour Commute' is associated with lower odds of severity (OR = 0.69, 18.0% vs 24.2%).
- Presence of 'Weekend Travel' is associated with lower odds of severity (OR = 0.97, 22.7% vs 23.2%).
- Presence of 'Traffic Signal Present' is associated with lower odds of severity (OR = 0.66, 17.4% vs 24.3%).
