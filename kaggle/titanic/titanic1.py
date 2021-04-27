import numpy as np
import pandas as pd

train = pd.read_csv('c:/data/kaggle/titanic/train.csv', index_col = 'PassengerId')
test = pd.read_csv('c:/data/kaggle/titanic/test.csv', index_col = 'PassengerId')

print(train.shape, test.shape) # (100000, 11) (100000, 10)

print(train.isna().sum())
test.isna().sum()

train.dtypes.unique()
test.dtypes.unique()

train.select_dtypes(include = ['object']).describe()
train.drop('Survived', axis = 1).select_dtypes(exclude = ['object']).describe()
target = train.Survived.copy()
train.drop('Survived', axis = 1).columns.equals(test.columns)

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

sns.set_style('whitegrid')

plt.figure(figsize = (16, 6))
sns.countplot(x = train.Survived, palette = 'Purples_r')

def plot_grid(data, fig_size, grid_size, plot_type, target = ''):
    """
    Custom function for plotting grid of plots.
    It takes: DataFrame of data, size of a grid, type of plots, string name of target variable;
    And it outputs: grid of plots.
    """
    fig = plt.figure(figsize = fig_size)
    if plot_type == 'histplot':
        for i, column_name in enumerate(data.select_dtypes(exclude = 'object').columns):
            fig.add_subplot(grid_size[0], grid_size[1], i + 1)
            plot = sns.histplot(data[column_name], kde = True, color = 'blueviolet', stat = 'count')
    if plot_type == 'boxplot':
        for i, column_name in enumerate(data.select_dtypes(exclude = 'object').columns):
            fig.add_subplot(grid_size[0], grid_size[1], i + 1)
            plot = sns.boxplot(x = data[column_name], color = 'blueviolet')
    if plot_type == 'countplot':
        target = data[target]
        for i, column_name in enumerate(data.drop(target.name, axis = 1).columns):
            fig.add_subplot(grid_size[0], grid_size[1], i + 1)
            plot = sns.countplot(x = data[column_name], hue = target, palette = 'Purples_r')
            plot.legend(loc = 'upper right', title = target.name)
    plt.tight_layout()

plot_grid(train.drop('Survived', axis = 1), (16, 6), (2,3), 'histplot')

pd.pivot_table(train, index = 'Survived', values = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass'], aggfunc = 'mean')

plot_grid(train.select_dtypes(exclude = 'object').drop(['Fare', 'Age'], axis = 1), (16, 6), (1, 3), 'countplot', 'Survived')

plt.figure(figsize = (16, 6))
sns.heatmap(train.corr(), 
            annot = True,
            fmt = '.2f',
            square = True,
            cmap = "Purples_r", 
            mask = np.triu(train.corr()))

plot_grid(train.drop('Survived', axis = 1), (16, 6), (2,3), 'boxplot')

plot_grid(pd.concat([train.select_dtypes(include = 'object').drop(['Name', 'Ticket', 'Cabin'], axis = 1), target], axis = 1), (16, 6), (2,1), 'countplot', 'Survived')

train.select_dtypes(include = 'object').nunique().sort_values(ascending = False)

train_test = pd.concat([train.drop('Survived', axis = 1), test], keys = ['train', 'test'], axis = 0)
missing_values = pd.concat([train_test.isna().sum(),
                            (train_test.isna().sum() / train_test.shape[0]) * 100], axis = 1, 
                            keys = ['Values missing', 'Percent of missing'])
missing_values.loc[missing_values['Percent of missing'] > 0].sort_values(ascending = False, by = 'Percent of missing').style.background_gradient('Purples')

train_cleaning = train.drop('Survived', axis = 1).copy()
test_cleaning = test.copy()

train_cleaning['Cabin'].fillna('none', inplace = True)
test_cleaning['Cabin'].fillna('none', inplace = True)

train_cleaning['Ticket'].fillna('none', inplace = True)
test_cleaning['Ticket'].fillna('none', inplace = True)

train_cleaning.loc[train_cleaning.Sex == 'male', 'Age'] = train_cleaning.loc[train_cleaning.Sex == 'male', 'Age'].fillna(train_cleaning.loc[train_cleaning.Sex == 'male', 'Age'].median())
train_cleaning.loc[train_cleaning.Sex == 'female', 'Age'] = train_cleaning.loc[train_cleaning.Sex == 'female', 'Age'].fillna(train_cleaning.loc[train_cleaning.Sex == 'female', 'Age'].median())
test_cleaning.loc[test_cleaning.Sex == 'male', 'Age'] = test_cleaning.loc[test_cleaning.Sex == 'male', 'Age'].fillna(train_cleaning.loc[train_cleaning.Sex == 'male', 'Age'].median())
test_cleaning.loc[test_cleaning.Sex == 'female', 'Age'] = test_cleaning.loc[test_cleaning.Sex == 'female', 'Age'].fillna(train_cleaning.loc[train_cleaning.Sex == 'female', 'Age'].median())

train_cleaning.loc[train_cleaning.Sex == 'male', 'Embarked'] = train_cleaning.loc[train_cleaning.Sex == 'male'].groupby('Pclass').Embarked.apply(lambda x: x.fillna(x.mode()[0]))
train_cleaning.loc[train_cleaning.Sex == 'female', 'Embarked'] = train_cleaning.loc[train_cleaning.Sex == 'female'].groupby('Pclass').Embarked.apply(lambda x: x.fillna(x.mode()[0]))

train_cleaning.loc[train_cleaning.Sex == 'male', 'Fare'] = train_cleaning.loc[train_cleaning.Sex == 'male'].groupby('Pclass').Fare.apply(lambda x: x.fillna(x.median()))
train_cleaning.loc[train_cleaning.Sex == 'female', 'Fare'] = train_cleaning.loc[train_cleaning.Sex == 'female'].groupby('Pclass').Fare.apply(lambda x: x.fillna(x.median()))
for i in train_cleaning.Pclass.unique():
    test_cleaning.loc[(test_cleaning.Pclass == i) & (test_cleaning.Sex == 'male'), 'Embarked'] = test_cleaning.loc[(test_cleaning.Pclass == i) & (test_cleaning.Sex == 'male'), 'Embarked'].fillna(train_cleaning.loc[(train_cleaning.Pclass == i) & (train_cleaning.Sex == 'male')].Embarked.mode()[0])
    test_cleaning.loc[(test_cleaning.Pclass == i) & (test_cleaning.Sex == 'female'), 'Embarked'] = test_cleaning.loc[(test_cleaning.Pclass == i) & (test_cleaning.Sex == 'female'), 'Embarked'].fillna(train_cleaning.loc[(train_cleaning.Pclass == i) & (train_cleaning.Sex == 'female')].Embarked.mode()[0])
    test_cleaning.loc[(test_cleaning.Pclass == i) & (test_cleaning.Sex == 'male'), 'Fare'] = test_cleaning.loc[(test_cleaning.Pclass == i) & (test_cleaning.Sex == 'male'), 'Fare'].fillna(train_cleaning.loc[(train_cleaning.Pclass == i) & (train_cleaning.Sex == 'male')].Fare.mode()[0])
    test_cleaning.loc[(test_cleaning.Pclass == i) & (test_cleaning.Sex == 'female'), 'Fare'] = test_cleaning.loc[(test_cleaning.Pclass == i) & (test_cleaning.Sex == 'female'), 'Fare'].fillna(train_cleaning.loc[(train_cleaning.Pclass == i) & (train_cleaning.Sex == 'female')].Fare.mode()[0])
    

# train_cleaning['Embarked'].fillna('none', inplace = True)
# test_cleaning['Embarked'].fillna('none', inplace = True)

train_test_cleaning = pd.concat([train_cleaning, test_cleaning], keys = ['train', 'test'], axis = 0)

train_test_cleaning['CabinLetter'] = train_test_cleaning.Cabin.str.split().apply(lambda x: x[-1][0].strip().lower() if x[0] != 'none' else np.nan)
train_test_cleaning['TicketLetters'] = train_test_cleaning.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower() 
                                                                        if len(x.split(' ')[:-1]) > 0 else np.nan)
# train_test_cleaning['CabinIsNull'] = train_test_cleaning.Cabin.apply(lambda x: 1 if x == 'none' else 0)
# train_test_cleaning['TicketIsNull'] = train_test_cleaning.Ticket.apply(lambda x: 1 if x == 'none' else 0)
# train_test_cleaning['EmbarkedIsNull'] = train_test_cleaning.Embarked.apply(lambda x: 1 if x == 'none' else 0)

train_cleaning_new = train_test_cleaning.xs('train').copy()
test_cleaning_new = train_test_cleaning.xs('test').copy()

train_cleaning_new.loc[train_cleaning_new.Sex == 'male'].groupby('Pclass').CabinLetter.apply(lambda x: x.value_counts().index[0])

train_cleaning_new.loc[train_cleaning_new.Sex == 'female'].groupby('Pclass').CabinLetter.apply(lambda x: x.value_counts().index[0])

train_cleaning_new.loc[train_cleaning_new.Sex == 'male'].groupby('Pclass').TicketLetters.apply(lambda x: x.value_counts().index[0])

train_cleaning_new.loc[train_cleaning_new.Sex == 'female'].groupby('Pclass').TicketLetters.apply(lambda x: x.value_counts().index[0])

# train_cleaning_new['CabinLetter'] = train_cleaning_new.groupby('Pclass')['CabinLetter'].apply(lambda x: x.fillna(x.mode()[0]))

train_cleaning_new.loc[train_cleaning_new.Sex == 'male', 'CabinLetter'] = train_cleaning_new.loc[train_cleaning_new.Sex == 'male'].groupby('Pclass')['CabinLetter'].apply(lambda x: x.fillna(x.mode()[0]))
train_cleaning_new.loc[train_cleaning_new.Sex == 'female', 'CabinLetter'] = train_cleaning_new.loc[train_cleaning_new.Sex == 'female'].groupby('Pclass')['CabinLetter'].apply(lambda x: x.fillna(x.mode()[0]))

train_cleaning_new.loc[train_cleaning_new.Sex == 'male', 'TicketLetters'] = train_cleaning_new.loc[train_cleaning_new.Sex == 'male'].groupby('Pclass')['TicketLetters'].apply(lambda x: x.fillna(x.mode()[0]))
train_cleaning_new.loc[train_cleaning_new.Sex == 'female', 'TicketLetters'] = train_cleaning_new.loc[train_cleaning_new.Sex == 'female'].groupby('Pclass')['TicketLetters'].apply(lambda x: x.fillna(x.mode()[0]))

for i in train_cleaning_new.Pclass.unique():
    test_cleaning_new.loc[(test_cleaning_new.Pclass == i) & (test_cleaning_new.Sex == 'male'), 'CabinLetter'] = test_cleaning_new.loc[(test_cleaning_new.Pclass == i) & (test_cleaning_new.Sex == 'male'), 'CabinLetter'].fillna(train_cleaning_new.loc[(train_cleaning_new.Pclass == i) & (train_cleaning_new.Sex == 'male')].CabinLetter.mode()[0])
    test_cleaning_new.loc[(test_cleaning_new.Pclass == i) & (test_cleaning_new.Sex == 'female'), 'CabinLetter'] = test_cleaning_new.loc[(test_cleaning_new.Pclass == i) & (test_cleaning_new.Sex == 'female'), 'CabinLetter'].fillna(train_cleaning_new.loc[(train_cleaning_new.Pclass == i) & (train_cleaning_new.Sex == 'female')].CabinLetter.mode()[0])
    
    test_cleaning_new.loc[(test_cleaning_new.Pclass == i) & (test_cleaning_new.Sex == 'male'), 'TicketLetters'] = test_cleaning_new.loc[(test_cleaning_new.Pclass == i) & (test_cleaning_new.Sex == 'male'), 'TicketLetters'].fillna(train_cleaning_new.loc[(train_cleaning_new.Pclass == i) & (train_cleaning_new.Sex == 'male')].TicketLetters.mode()[0])
    test_cleaning_new.loc[(test_cleaning_new.Pclass == i) & (test_cleaning_new.Sex == 'female'), 'TicketLetters'] = test_cleaning_new.loc[(test_cleaning_new.Pclass == i) & (test_cleaning_new.Sex == 'female'), 'TicketLetters'].fillna(train_cleaning_new.loc[(train_cleaning_new.Pclass == i) & (train_cleaning_new.Sex == 'female')].TicketLetters.mode()[0])

    
train_test_cleaning = pd.concat([train_cleaning_new, test_cleaning_new], keys = ['train', 'test'], axis = 0)

train.loc[:, ['Fare', 'Age']].select_dtypes(exclude = ['object']).describe()

train_test_cleaning['CabinNumbers'] = train_test_cleaning.Cabin.apply(lambda x: int(x[1:]) if x != 'none' else -1)

train_test_cleaning['TicketNumbers'] = train_test_cleaning.Ticket.apply(lambda x: int(x) if x.isnumeric() else -1 if x == 'none' else int(x.split(' ')[-1]) if (x.split(' ')[-1]).isnumeric() else 0)
train_test_cleaning['TicketNumbersGroup'] = train_test_cleaning['TicketNumbers'].apply(lambda x: 0 if (x == -1)
                                                                                       else 1 if (x > -1 and x <= 100000)
                                                                                       else 2 if (x > 100000 and x <= 260000)                                                                    
                                                                                       else 3 if (x > 260000 and x <= 380000)
                                                                                       else 4 if (x > 380000 and x <= 538000)
                                                                                       else 5)

train_test_cleaning['TicketIsNumeric'] = train_test_cleaning.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)

train_test_cleaning['FamilySize'] = train_test_cleaning.SibSp + train_test_cleaning.Parch + 1
train_test_cleaning['FamilySize'] = train_test_cleaning['FamilySize'].apply(lambda x: 0 if (x == 1)
                                                                            else 1 if (x == 2 or x == 3)
                                                                            else 2)
train_test_cleaning['IsAlone'] = train_test_cleaning['FamilySize'].apply(lambda x: 0 if (x == 1) else 1)

# train_test_cleaning['AgeGroup'] = train_test_cleaning['Age'].apply(lambda x: '0 'if (x < 25) 
#                                                                    else '1' if (x >= 25 and x < 39)                                                                    
#                                                                    else '2' if (x >= 39 and x < 53)
#                                                                    else '3')

train_test_cleaning['AgeGroup'] = train_test_cleaning['Age'].apply(lambda x: '0 'if (x < 10) 
                                                                   else '1' if (x >= 10 and x < 20)                                                                    
                                                                   else '2' if (x >= 20 and x < 30)
                                                                   else '3' if (x >= 30 and x < 40)
                                                                   else '4' if (x >= 40 and x < 50)
                                                                   else '5' if (x >= 50 and x < 60)
                                                                   else '6' if (x >= 60 and x < 70)
                                                                   else '7' if (x >= 70 and x < 80)
                                                                   else '8')

train_test_cleaning['FareGroup'] = train_test_cleaning['Fare'].apply(lambda x: '0 'if (x < 10.04) 
                                                                     else '1' if (x >= 10.04 and x < 24.46)  
                                                                     else '2' if (x >= 24.46 and x < 33.5)                                                            
                                                                     else '3')

train_test_cleaning['TicketLettersGroup'] = train_test_cleaning.TicketLetters.apply(lambda x: 0 if x == 'pc' 
                                                                                    else 3 if x in ['stono', 'stono2', 'sotono2', 'stonoq', 'aq3']
                                                                                    else 2 if x in ['sotonoq', 'fa', 'a5', 'ca', 'fcc', 'scow', 'casoton', 'a4', 'wc', 'swpp', 'c']                                                                
                                                                                    else 1)

# train_test_cleaning['Surname'] = train_test_cleaning['Name'].apply(lambda x: x.split(',')[0].lower())

train_test_cleaning['Embarked'] = train_test_cleaning['Embarked'].str.lower()

from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p

lamb = boxcox_normmax(train_test_cleaning.loc['train', 'Fare'] + 1)
train_test_cleaning.loc['train', 'Fare'] = boxcox1p(train_test_cleaning.loc['train', 'Fare'], lamb).values
train_test_cleaning.loc['test', 'Fare'] = boxcox1p(train_test_cleaning.loc['test', 'Fare'], lamb).values

train_cleaning_target_cleaned = pd.concat([train_test_cleaning.xs('train'), target], axis = 1)

print(f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'CabinLetter', values = 'Name', aggfunc ='count')} \n\n" +
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', values = 'TicketNumbers', aggfunc = (lambda x: x.mode()[0]))} \n\n" +
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'TicketIsNumeric', values = 'Name', aggfunc ='count')} \n\n" +
      
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'AgeGroup', values = 'Name', aggfunc ='count')} \n\n" +
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'FareGroup', values = 'Name', aggfunc ='count')} \n\n" +
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'TicketLettersGroup', values = 'Name', aggfunc ='count')} \n\n" +
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'TicketNumbersGroup', values = 'Name', aggfunc ='count')} \n\n" +
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'IsAlone', values = 'Name', aggfunc ='count')} \n\n" +
      
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'FamilySize', values = 'Name', aggfunc ='count')}")

pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'TicketLetters', values = 'Name', aggfunc = 'count')

train_cleaning_target_cleaned.select_dtypes(include = 'object').nunique().sort_values(ascending = False)

plot_grid(train_cleaning_target_cleaned.loc[:,['Age', 'Fare', 'TicketNumbers', 'CabinNumbers']], (16, 6), (2, 3), 'histplot')

plot_grid(train_cleaning_target_cleaned.drop(['Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'TicketNumbers', 'TicketLetters', 'CabinNumbers'],
                                             axis = 1), (16, 10), (5, 3), 'countplot', 'Survived')

pd.crosstab(index = train_cleaning_target_cleaned.TicketLetters , columns= train_cleaning_target_cleaned.Survived, normalize = 'index' ). \
sort_values(by = 1).plot.bar(figsize = (15, 7), stacked = True, color = {0: 'grey', 
                                                                         1: 'purple'})
plt.axhline(y = 0.8, color = 'r', linestyle = '-')
plt.axhline(y = 0.65, color = 'g', linestyle = '-')

from matplotlib import ticker
# 'Age', 'Fare', 'TicketNumbers', 'CabinNumbers'
fig, axs = plt.subplots(4, 1, figsize = (16, 16))
sns.histplot(hue = train_cleaning_target_cleaned.Survived, x = train_cleaning_target_cleaned.Age, palette = {0 : 'black', 1 : 'purple'}, ax = axs[0])
axs[0].set_title('Age distribution')
sns.histplot(hue = train_cleaning_target_cleaned.Survived, x = train_cleaning_target_cleaned.Fare, palette = {0 : 'black', 1 : 'purple'}, ax = axs[1])
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(25))
axs[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
axs[1].set_title('Fare distribution')
sns.histplot(hue = train_cleaning_target_cleaned.Survived, x = train_cleaning_target_cleaned.TicketNumbers, palette = {0 : 'black', 1 : 'purple'}, ax = axs[2])
axs[2].set_title('TicketNumbers distribution')
sns.histplot(hue = train_cleaning_target_cleaned.Survived, x = train_cleaning_target_cleaned.CabinNumbers, palette = {0 : 'black', 1 : 'purple'}, ax = axs[3])
axs[3].set_title('CabinNumbers distribution')
plt.tight_layout()


plt.figure(figsize = (16,10))
sns.heatmap(train_cleaning_target_cleaned.corr(),
            annot = True,
            annot_kws = {"size": 13},
            fmt = '.2f',
            square = True,
            cmap = "Purples_r",
            mask = np.triu(train_cleaning_target_cleaned.corr()))

to_drop = ['Name',
           'Ticket',
           'Cabin']

train_test_cleaned = train_test_cleaning.drop(to_drop, axis = 1).copy()

label_cols = ['AgeGroup', 'FamilySize', 'TicketLettersGroup', 'Pclass', 'IsAlone', 'TicketIsNumeric', 'TicketNumbersGroup']
onehot_cols = ['CabinLetter', 'Embarked', 'Sex']
numerical_cols = ['SibSp', 'Parch', 'Fare']

# label_cols = ['AgeGroup', 'FamilySize', 'TicketLettersGroup', 'Sex', 'Pclass', 'IsAlone', 'TicketIsNumeric', 'TicketNumbersGroup', 'CabinIsNull', 'TicketIsNull', 'EmbarkedIsNull']
# onehot_cols = ['CabinLetter', 'Embarked']
# numerical_cols = ['SibSp', 'Parch', 'Fare']

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# One-hot encoding
train_test_onehot = pd.get_dummies(train_test_cleaned[onehot_cols])
X_train_full_onehot, X_test_onehot = train_test_onehot.xs('train').reset_index(), train_test_onehot.xs('test').reset_index()

X_train_full, X_test = train_test_cleaned.xs('train'), train_test_cleaned.xs('test')
# Label encoding
X_train_full_labeled = pd.DataFrame()
X_test_labeled = pd.DataFrame()
for col in label_cols:
    encoder = LabelEncoder()
    encoder.fit(X_train_full[col])
    
    encoded_train = pd.Series(encoder.transform(X_train_full[col]), name = col)
    X_train_full_labeled = pd.concat([X_train_full_labeled, encoded_train], axis = 1)
    
    encoded_test = pd.Series(encoder.transform(X_test[col]), name = col)
    X_test_labeled = pd.concat([X_test_labeled, encoded_test], axis = 1)
# Numerical features scaling
scaler = StandardScaler()
scaler.fit(X_train_full[numerical_cols])
X_train_full_scaled = pd.DataFrame(scaler.transform(X_train_full[numerical_cols]), columns = numerical_cols)
X_test_scaled = pd.DataFrame(scaler.transform(X_test[numerical_cols]), columns = numerical_cols)
# Concatenating it all together
X_train_full = pd.concat([X_train_full_onehot, 
                          X_train_full_labeled, 
                          X_train_full_scaled], axis = 1)
X_train_full.set_index('PassengerId', inplace = True)
X_test = pd.concat([X_test_onehot, 
                    X_test_labeled, 
                    X_test_scaled], axis = 1)
X_test.set_index('PassengerId', inplace = True)

y_train_full = target

# train_test_cleaned_male = train_test_cleaned.loc[train_test_cleaned.Sex == 'male'].copy()
# train_test_cleaned_female = train_test_cleaned.loc[train_test_cleaned.Sex == 'female'].copy()
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # Male
# # One-hot encoding
# train_test_onehot = pd.get_dummies(train_test_cleaned_male[onehot_cols])
# X_train_full_onehot, X_test_onehot = train_test_onehot.xs('train').reset_index(), train_test_onehot.xs('test').reset_index()

# X_train_full, X_test = train_test_cleaned_male.xs('train'), train_test_cleaned_male.xs('test')
# # Label encoding
# X_train_full_labeled = pd.DataFrame()
# X_test_labeled = pd.DataFrame()
# for col in label_cols:
#     encoder = LabelEncoder()
#     encoder.fit(X_train_full[col])
    
#     encoded_train = pd.Series(encoder.transform(X_train_full[col]), name = col)
#     X_train_full_labeled = pd.concat([X_train_full_labeled, encoded_train], axis = 1)
    
#     encoded_test = pd.Series(encoder.transform(X_test[col]), name = col)
#     X_test_labeled = pd.concat([X_test_labeled, encoded_test], axis = 1)
# # Numerical features scaling
# scaler = StandardScaler()
# scaler.fit(X_train_full[numerical_cols])
# X_train_full_scaled = pd.DataFrame(scaler.transform(X_train_full[numerical_cols]), columns = numerical_cols)
# X_test_scaled = pd.DataFrame(scaler.transform(X_test[numerical_cols]), columns = numerical_cols)
# # Concatenating it all together
# X_train_full_male = pd.concat([X_train_full_onehot, 
#                           X_train_full_labeled, 
#                           X_train_full_scaled], axis = 1)
# X_train_full_male.set_index('PassengerId', inplace = True)
# X_test_male = pd.concat([X_test_onehot, 
#                     X_test_labeled, 
#                     X_test_scaled], axis = 1)
# X_test_male.set_index('PassengerId', inplace = True)
# X_train_full_male
# y_train_full_male = target.loc[target.index.isin(X_train_full_male.index)].copy()
# y_train_full_male
# # Female
# # One-hot encoding
# train_test_onehot = pd.get_dummies(train_test_cleaned_female[onehot_cols])
# X_train_full_onehot, X_test_onehot = train_test_onehot.xs('train').reset_index(), train_test_onehot.xs('test').reset_index()

# X_train_full, X_test = train_test_cleaned_female.xs('train'), train_test_cleaned_female.xs('test')
# # Label encoding
# X_train_full_labeled = pd.DataFrame()
# X_test_labeled = pd.DataFrame()
# for col in label_cols:
#     encoder = LabelEncoder()
#     encoder.fit(X_train_full[col])
    
#     encoded_train = pd.Series(encoder.transform(X_train_full[col]), name = col)
#     X_train_full_labeled = pd.concat([X_train_full_labeled, encoded_train], axis = 1)
    
#     encoded_test = pd.Series(encoder.transform(X_test[col]), name = col)
#     X_test_labeled = pd.concat([X_test_labeled, encoded_test], axis = 1)
# # Numerical features scaling
# scaler = StandardScaler()
# scaler.fit(X_train_full[numerical_cols])
# X_train_full_scaled = pd.DataFrame(scaler.transform(X_train_full[numerical_cols]), columns = numerical_cols)
# X_test_scaled = pd.DataFrame(scaler.transform(X_test[numerical_cols]), columns = numerical_cols)
# # Concatenating it all together
# X_train_full_female = pd.concat([X_train_full_onehot, 
#                           X_train_full_labeled, 
#                           X_train_full_scaled], axis = 1)
# X_train_full_female.set_index('PassengerId', inplace = True)
# X_test_female = pd.concat([X_test_onehot, 
#                     X_test_labeled, 
#                     X_test_scaled], axis = 1)
# X_test_female.set_index('PassengerId', inplace = True)
# X_train_full_female
# y_train_full_female = target.loc[target.index.isin(X_train_full_female.index)].copy()
# y_train_full_female