import pandas as pd
import pandas_profiling
aa = pd.read_csv('c:/data/music/3s_data.csv')
pr=aa.profile_report()

pr.to_file('c:/data/music/pr_report_3s_data.html')