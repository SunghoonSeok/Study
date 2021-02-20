import pandas as pd
import pandas_profiling
aa = pd.read_csv('c:/data/music/df1_mean_3s.csv')
pr=aa.profile_report()

pr.to_file('c:/data/music/pr_report_mean_3s.html')