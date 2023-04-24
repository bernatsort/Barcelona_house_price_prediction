def explore_data(df: pd.DataFrame) -> None:
    """
    Takes in a pandas DataFrame and performs exploratory data analysis.

    Parameters:
    -----------
    df: pandas DataFrame
        The DataFrame to be analyzed.

    Returns:
    --------
    None
    """

    # Print the dimensions of the DataFrame
    print(f"The dataset includes {df.shape[0]} instances (rows) and {df.shape[1]} variables (columns).\n")

    # Display the first few rows of the DataFrame
    display(df.head())
    print("\n")

    # Print the column information for the DataFrame
    print(df.info())
    print("\n")
    
    # Check missing values
    print(f"Missing values: \n{df.isna().sum()}")
    print("\n")

    # Select only the numeric features
    numeric_vars = df.select_dtypes(include=['int64', 'float64']).columns

    # Calculate the descriptive statistics for the numeric variables
    display(df[numeric_vars].describe())
    
    # Check duplicates
    print(f"\n Number of duplicates: {df.duplicated().sum()}")


def plot_histogram(data: pd.DataFrame, variable: str) -> None:
    """
    Creates a histogram of a given variable in a given DataFrame.

    Parameters:
    -----------
    data : pandas DataFrame
        The DataFrame containing the target variable to be plotted.
    target_variable : str
        The name of the target variable to be plotted.

    Returns:
    --------
    None
    """

    fig, axs = plt.subplots(figsize=(10, 5))
    sns.histplot(data[variable])
    plt.title(f'Distribution of {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuency')
    plt.show()


def plot_boxplot(data: pd.DataFrame, target_variable: str, show_outliers: bool = False) -> None:
    """
    Creates a boxplot of a given variable in a given DataFrame.

    Parameters:
    -----------
    data : pandas DataFrame
        The DataFrame containing the target variable to be plotted.
    target_variable : str
        The name of the target variable to be plotted.
    show_outliers : bool, optional (default=False)
        Whether to show or hide the outliers in the boxplot.

    Returns:
    --------
    None
    """

    sns.set(rc={"figure.figsize": (10, 5)}, style='whitegrid')  
    if show_outliers:
        b = sns.boxplot(x=data[target_variable],
                      palette ='pastel') 
    else:
        b = sns.boxplot(x=data[target_variable],
                      palette ='pastel', 
                      showfliers = False)

    b.set_xlabel(target_variable, fontsize=14)
    b.set_title(f"Boxplot of {target_variable} ({'showing outliers' if show_outliers else 'without showing outliers'})", fontsize=16)

    for patch in b.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .4))

    plt.show()


def scatter_plots(df: pd.DataFrame, 
                  target: str,
                  numeric_vars: List[str], 
                  figsize: Tuple[int, int]=(15,20)) -> None:
    """
    Creates scatter plots for each numeric variable in the given DataFrame (excluding the target variable).
    
    Args:
    - df: Pandas DataFrame.
    - target: Name of the target variable.
    - numeric_vars: List of names of the numeric variables to plot.
    - figsize: Tuple indicating the size of the figure to create.
    
    Returns: None.
    """
    
    # Create subplots with number of rows and columns based on number of numeric variables
    n_vars = len(numeric_vars)
    n_rows = (n_vars - 1) // 2 + 1
    fig, axs = plt.subplots(n_rows, 2, figsize=figsize)
    
    # Flatten the matrix of subplots for easier iteration
    axs = axs.flatten()
    
    # Iterate over each variable and create a scatter plot on the corresponding subplot
    for i, col in enumerate(numeric_vars):
        ax = axs[i]
        ax.scatter(df[col], df[target], alpha=0.2)
        ax.set_title(f"Relationship between {col} and {target}")
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        
    # Adjust the space between the subplots and remove empty plot if number of variables is odd
    plt.tight_layout()
    if n_vars % 2 != 0:
        axs[-1].remove()
        
    # Show the plot
    plt.show()


def find_constant_variables(df: pd.DataFrame) -> list:
    """
    Returns a list of column names that have the same value in all rows of the dataframe
    
    Parameters:
    df (pandas dataframe): input dataframe
    
    Returns:
    list: list of column names that have the same value in all rows
    """
    constant_vars = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_vars.append(col)
    return constant_vars


def count_null_values(df: pd.DataFrame) -> None:
    """
    Print the names of any columns in a Pandas DataFrame that contain null values, and the number of null values in each column.
    Also plots a bar chart showing the number of null values for each variable in the DataFrame.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame to check for null values.
    """
    null_counts = df.isnull().sum()
    null_vars = null_counts[null_counts != 0].index.tolist()
    
    if null_vars:
        print("Variables con valores nulos:\n")
        for var in null_vars:
            print(f"\t- '{var}':  {null_counts[var]}")

        # plot the number of null values for each variable
        fig, ax = plt.subplots(figsize=(8, 4))
        null_counts[null_vars].plot(kind='bar', ax=ax)
        ax.set_title('Number of missing values per variable')
        ax.set_xlabel('Variable')
        ax.set_ylabel('Count')
        plt.show()

    else:
        print("No se encontraron valores nulos en el dataset.\n")

def get_binary_cols(df: pd.DataFrame) -> list[str]:
    """
    Returns a list with the names of the binary variables in a Pandas DataFrame.
    
    Parameters:
        df: Pandas DataFrame to check for binary variables.
        
    Returns:
        A list with the names of the binary variables in the DataFrame.
    """
    binary_cols = [col for col in df if (len(df[col].value_counts()) > 0) and all(df[col].value_counts().index.isin([0, 1]))]

    return binary_cols


def replace_binary_cols_nulls_by_zero(df: pd.DataFrame) -> None:
    """
    Replaces missing values in binary columns of a pandas DataFrame with 0.
    
    Parameters:
        df: Input pandas DataFrame
        
    Returns: 
        None    
    """
    binary_cols = get_binary_cols(df)
    df.loc[:, binary_cols] = df.loc[:, binary_cols].fillna(0)


def check_value_counts(df: pd.DataFrame, cols: List[str]) -> None:
    """
    Print the value counts of the specified columns in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to check value counts for.
    cols (List[str]): A list of column names to check value counts for.
    
    Returns:
    None
    """
    for col in cols:
        print(f"Value counts for column '{col}':")
        print(df[col].value_counts())
        print()

def convert_object_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns in a pandas dataframe to category data type.
    
    Parameters:
    df (pd.DataFrame): Input pandas dataframe
    
    Returns:
    pd.DataFrame: A copy of the input dataframe with object columns converted to category data type.
    """    
    # loop through columns and convert object columns to category
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    
    return df


def plot_corr_barchart(df1: pd.DataFrame, target: str,  n: int = 10) -> None:
    """
    Plots a color-gradient bar chart showing top n correlations between features
    Args:
        df1 (pd.DataFrame): the dataframe to plot
        n (int): number of top n correlations to plot
    Returns:
        None
    Sources: 
    https://typefully.com/levikul09/j6qzwR0
    https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
    """
    print(f"Correlation between numeric features (excluding the target variable: {target})\n")

    # drop target
    df1 = df1.drop(columns=target)

    # select only the numeric features
    useful_columns =  df1.select_dtypes(include=['int64', 'float64']).columns

    def get_redundant_pairs(df):
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0,df.shape[1]):
            for j in range(0,i+1):
                pairs_to_drop.add((cols[i],cols[j]))
        return pairs_to_drop

    def get_correlations(df,n=n):
        au_corr = df.corr(method = 'spearman').unstack() # spearman used because not all data is normalized
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels = labels_to_drop).sort_values(ascending=False)
        top_n = au_corr[0:n]    
        bottom_n =  au_corr[-n:]
        top_corr = pd.concat([top_n, bottom_n])
        return top_corr

    corrplot = get_correlations(df1[useful_columns])


    fig, ax = plt.subplots(figsize=(20,15))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax =1)
    colors = [plt.cm.RdYlGn(norm(c)) for c in corrplot.values]
    
    # n*(n-1)/2-->  maximum value of "n" for which there are no repeated correlations
    # n = number of numerical variables = len(useful_columns)
    num_corr = (len(useful_columns)*(len(useful_columns)-1)/2)
    print(f"Max number of correlations (n): {num_corr/2}\n")
    
    print(corrplot)

    corrplot.plot.barh(color=colors)


def plot_corr_vs_target(df1: pd.DataFrame, target: str, n: int) -> None:
    """
    Plots a color-gradient bar chart showing top n correlations between features and target
    Args:
        target (str): the name of the target column
        df1 (pd.DataFrame): the dataframe to plot
        n (int): number of top n correlations to plot
    Returns:
        None
    """
    print(f"Correlation between numeric features and the target variable {target}\n")

    # select the number of numeric features
    useful_columns =  len(df1.select_dtypes(include=['int64', 'float64']).columns)
    print(f"Max number of correlations (n): {useful_columns/2}\n")
    
    target_series = df1[target]
    
    x = df1.corrwith(target_series, method = 'spearman', numeric_only=True).sort_values(ascending=False)
    top_n = x[0:n]    
    bottom_n =  x[-n:]
    top_corr = pd.concat([top_n, bottom_n])
    x = top_corr
    print(x)

    fig, ax = plt.subplots(figsize=(8,4))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax =1)
    colors = [plt.cm.RdYlGn(norm(c)) for c in x.values]
    x.plot.barh(color=colors)


def calculate_correlation(df: pd.DataFrame, method: str, figsize: tuple = (6, 4)) -> pd.DataFrame:
    """
    Calculates the correlation matrix between numeric features in a pandas DataFrame using a specified correlation method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to calculate the correlation matrix for.
    method (str): The correlation method to use. Valid options are 'pearson', 'spearman', and 'kendall'.
    figsize (tuple): The size of the heatmap plot. Default is (6, 4).
    
    Returns:
    pd.DataFrame: The correlation matrix between numeric features in the DataFrame.
    """
    # select only the numeric features
    numeric_features = df.select_dtypes(include=['int64', 'float64'])
    X = numeric_features
    
    # scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # calculate correlation matrix
    if method == 'pearson':
        corr_matrix = pd.DataFrame(X_scaled).corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = pd.DataFrame(X_scaled).corr(method='spearman')
    elif method == 'kendall':
        corr_matrix = pd.DataFrame(X_scaled).corr(method='kendall')
    else:
        raise ValueError("Invalid correlation method specified. Valid options are 'pearson', 'spearman', and 'kendall'.")
    
    # set column names of correlation matrix
    corr_matrix.columns = numeric_features.columns
    corr_matrix.index = numeric_features.columns
    
    # plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(f"Correlation Matrix ({method.capitalize()})")
    plt.show()
    
    return corr_matrix


def check_normality(pvalue):
    if pvalue > 0.05:
        print('Since the p-value > 0.05, we fail to reject the null hypothesis i.e. we assume the distribution of our variable is normal/gaussian.')
    else:
        print('Since p-value ≤ 0.05, we reject the null hypothesis i.e. we assume the distribution of our variable is not normal/gaussian.')


def convert_object_to_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns in a pandas dataframe to category data type.
    
    Parameters:
    df (pd.DataFrame): Input pandas dataframe
    
    Returns:
    pd.DataFrame: A copy of the input dataframe with object columns converted to category data type.
    """    
    # loop through columns and convert object columns to category
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    
    return df

def convert_var_to_category(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Convert specified columns in a pandas dataframe to category data type.
    
    Parameters:
    df (pd.DataFrame): Input pandas dataframe
    cols (List[str]): List of column names to convert
    
    Returns:
    pd.DataFrame: A copy of the input dataframe with specified columns converted to category data type.
    """    
    # loop through specified columns and convert columns to category           
    # convertimos las variables numéricas que son 0 y 1 a categóricas (binarias) 
    for i in range(0,len(cols)):
        # primero las convertimos a int64 para eliminar los decimales: 0 y 1
        df[cols[i]] = df[cols[i]].astype("int64")
        # luego las convertimos a 'category': '0' y '1'
        df[cols[i]] = df[cols[i]].astype("category")
        
    return 


def create_violinplots(df: pd.DataFrame, 
                       cat_cols: List[str], 
                       target_var: str, 
                       plot_width: int = 8, 
                       plot_height: int = 6) -> None:
    """
    Creates violin plots for the target variable across categorical variables in the given pandas DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        cat_cols (List[str]): A list of the names of the categorical columns.
        target_var (str): The name of the target variable.
        plot_width (int): The width of the plot in inches. Default is 8.
        plot_height (int): The height of the plot in inches. Default is 6.


    Returns:
        None.
    """
    for col in cat_cols:
        #  if num_categories is greater than 10, the x-axis label rotation is set to 90 degrees, otherwise it is set to 0.
        num_categories = len(df[col].unique())
        if num_categories > 10:
            x_label_rotation = 90
        else: 
            x_label_rotation = 0    
            
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        sns.violinplot(x=col, y=target_var, data=df, ax=ax)
        # show individual points
        sns.stripplot(x=col, y=target_var, data=df, jitter=False, color='black', size=4, alpha=0.5)
        # visualize the mean value on the violin plot
        sns.pointplot(x=col, y=target_var, data=df, color='red', ax=ax, errorbar=None)
        plt.title(f"Distribution of {target_var} by {col}")
        plt.xlabel(col)
        plt.ylabel(target_var)
        
        if x_label_rotation != 0:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=x_label_rotation)
        sns.despine()
        plt.show()


def create_boxplots(df: pd.DataFrame, 
                    cat_cols: List[str], 
                    target_var: str, 
                    plot_width: int = 8, 
                    plot_height: int = 6) -> None:
    """
    Creates boxplots for the target variable across categorical variables in the given pandas DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        cat_cols (List[str]): A list of the names of the categorical columns.
        target_var (str): The name of the target variable.
        plot_width (int): The width of the plot in inches. Default is 8.
        plot_height (int): The height of the plot in inches. Default is 6.


    Returns:
        None.
    """
    for col in cat_cols:
        #  if num_categories is greater than 10, the x-axis label rotation is set to 90 degrees, otherwise it is set to 0.
        num_categories = len(df[col].unique())
        if num_categories > 10:
            x_label_rotation = 90
        else: 
            x_label_rotation = 0    
            
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        sns.boxplot(x=col, y=target_var, data=df, ax=ax)
        # show individual points
        sns.stripplot(x=col, y=target_var, data=df, jitter=False, color='black', size=4, alpha=0.5)
        # visualize the mean value on the box plot
        sns.pointplot(x=col, y=target_var, data=df, color='red', ax=ax, markers="D", linestyles='--')
        # visualize the median value on the box plot
        median_val = df.groupby(col)[target_var].median().sort_values()
        #for i, val in enumerate(median_val):
            #ax.text(i, val, f"{val:.2f}", horizontalalignment='center', verticalalignment='bottom', fontdict={'size': 10})
        plt.title(f"Distribution of {target_var} by {col}")
        plt.xlabel(col)
        plt.ylabel(target_var)
        
        if x_label_rotation != 0:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=x_label_rotation)
        sns.despine()
        plt.show()


def perform_pca(df: pd.DataFrame, target_variable: str, n_components: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Perform principal component analysis (PCA) on a DataFrame with numeric features and plot the variance explained by each 
    principal component and the loadings for each principal component. Returns the loadings for each component and the transformed data.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with numeric features.
    target_variable : str
        Name of the target variable column to be removed from the DataFrame.
    n_components : int, optional
        Number of principal components to compute. Default is 3.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing the loadings for each component and the transformed data.
        
    Example:
    --------
    # perform PCA on the DataFrame
    loadings, transformed_data = perform_pca(df, 'target_variable', 2)
    """

    # remove the target variable from the DataFrame 
    df = df.drop(columns=target_variable)
    
    # select only the numeric features and drop na values 
    numeric_features = df.select_dtypes(include=['int64', 'float64']).dropna()    
    X = numeric_features
    
    # scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # apply PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # calculate the variance explained by each component
    variance = pd.DataFrame({'variance': pca.explained_variance_ratio_*100, # *100: Percentage of explained variances
                             'PC': ['PC{}'.format(i) for i in range(1, n_components+1)]})
    variance = variance.sort_values('variance', ascending=False)

    # plot the variance explained by each component
    sns.barplot(x='PC', y='variance', data=variance, color='blue')
    plt.title('Variance explained by each principal component')
    plt.xlabel('Principal Component')
    plt.ylabel('Percentage of explained variances') 
    plt.show()

    # get the loadings for each component
    loadings = pd.DataFrame(pca.components_.T, columns=['PC{}'.format(i) for i in range(1, n_components+1)], index=X.columns)

    # plot the loadings for each principal component
    n_rows = (n_components + 1) // 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(15, 6 * n_rows), sharey=False, gridspec_kw={'hspace': 0.4, 'wspace': 0.5})
    
    for i in range(n_components):
        row = i // 2
        col = i % 2
        loadings_pc = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
        sns.barplot(x=loadings_pc.values, y=loadings_pc.index, ax=axs[row, col], color='blue', order=loadings_pc.index)
        axs[row, col].set_title(f'Loadings for PC{i+1}')
        axs[row, col].set_xlabel('Loading Value')
        axs[row, col].set_ylabel('Variable')
        
    # adjust the subplots if n_components is odd to remove an empty subplot
    if n_components % 2 != 0:
        axs[n_rows-1, 1].remove()

    plt.show()

    # return the transformed data and the loadings
    transformed_data = pd.DataFrame(pca.transform(X_scaled), columns=['PC{}'.format(i) for i in range(1, n_components+1)])
    return loadings, transformed_data


def mutual_info_regression_analysis(df: pd.DataFrame, 
                                     target_col: str, 
                                     figsize: tuple = (10, 15), 
                                     encoder: str = None, 
                                     ohe_cols: List[str] = None, 
                                     le_cols: List[str] = None) -> pd.DataFrame:
    """
    Performs mutual information analysis between the predictors and target variable in the given dataframe and
    returns a sorted dataframe containing feature names and their mutual information scores.
    
    Parameters:
        df: A pandas dataframe containing the predictors and target variable.
        target_col: A string indicating the name of the target variable column.
        figsize: A tuple indicating the size of the output plot (default is (10, 5)).
        encoder: An optional string indicating which encoder to use, either "ohe" for One-Hot Encoding or "le" for Label Encoding.
        ohe_cols: An optional list of column names to be one-hot encoded.
        le_cols: An optional list of column names to be label encoded.

    Returns: 
        A pandas dataframe containing feature names and their mutual information scores.
    """
    # drop rows with na
    df = df.dropna(axis=1)

    # Separate the predictors and target variable
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Apply one-hot encoding if specified
    if ohe_cols:
        categorical_cols = list(set(ohe_cols) & set(X.columns))
        if categorical_cols:
            ohe = OneHotEncoder()
            ohe.fit(X[categorical_cols])
            encoded_cols = ohe.get_feature_names_out(categorical_cols)
            X_encoded = pd.concat([X.drop(categorical_cols, axis=1), 
                                   pd.DataFrame(ohe.transform(X[categorical_cols]).toarray(), 
                                                columns=encoded_cols)], axis=1)
            X = X_encoded
   # Apply encoding if specified
    elif encoder == "ohe":
    # One-hot encode categorical variables
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        if categorical_cols:
            ohe = OneHotEncoder()
            ohe.fit(X[categorical_cols])
            encoded_cols = ohe.get_feature_names_out(categorical_cols)
            X_encoded = pd.concat([X.drop(categorical_cols, axis=1), 
                                   pd.DataFrame(ohe.transform(X[categorical_cols]).toarray(), 
                                                columns=encoded_cols)], axis=1)
            X = X_encoded
    
    # Apply label encoding if specified
    if le_cols:
        categorical_cols = list(set(le_cols) & set(X.columns))
        if categorical_cols:
            le = LabelEncoder()
            X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))
    elif encoder == "le":
        # Label encode categorical variables
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        if categorical_cols:
            le = LabelEncoder()
            X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

    # Delete categorical vars not specified to encode
    cat_vars = X.select_dtypes(include=['category', 'object']).columns.tolist()
    if cat_vars: # if not empty
        # Remove columns with categorical variables
        categorical_cols_in_data = set(cat_vars).intersection(set(X.columns))
        X = X.drop(categorical_cols_in_data, axis=1)

    
    # Determine which predictors are discrete: all discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int

    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    # Plot mutual information scores
    fig, ax = plt.subplots(figsize=figsize)
    mi_scores.sort_values().plot.barh(ax=ax)
    ax.set(title="Mutual Information Scores", xlabel="MI Score", ylabel="Features")
    plt.show()

    return mi_scores


def perform_anova_kruskall(df: pd.DataFrame, target_col: str, normal_data: bool = True) -> pd.DataFrame:
    """
    Performs ANOVA or Kruskal-Wallis analysis for each categorical variable in the dataframe.
    
    Parameters:
        df: A pandas dataframe containing the predictors and target variable.
        target_col: A string indicating the name of the target variable column.
        normal_data: A boolean indicating whether the data is normally distributed (default True).
    
    Returns: 
        A pandas dataframe containing ANOVA or Kruskal-Wallis results for each categorical variable.
    """
    
    # Select categorical columns
    cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    
    # Perform ANOVA or Kruskal-Wallis for each categorical variable
    result_list = []
    for col in cat_cols:
        if normal_data:
            model = ols(f"{target_col} ~ {col}", data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            result_list.append(anova_table)
        else:
            groups = [group[target_col].values for name, group in df.groupby(col)]
            groups = [g for g in groups if len(g)>0] # remove empty groups
            if len(groups)>1:
                stat, p = stats.kruskal(*groups)
                table = pd.DataFrame({'Group': [col], 'Statistic': [stat], 'p-value': [p]})
                result_list.append(table)
    
    # Combine the results into a single dataframe
    result_df = pd.concat(result_list, axis=0)
    
    # Filter out the residual rows
    result_df = result_df[result_df.index != "Residual"]
    
    return result_df


def perform_extra_trees_regressor(df: pd.DataFrame, 
                                  target_col: str, 
                                  figsize: tuple = (10, 15), 
                                  encoder: str = None, 
                                  ohe_cols: List[str] = None, 
                                  le_cols: List[str] = None,
                                  n_estimators: int = 100, 
                                  criterion: str ='squared_error', 
                                  max_features: float = 1.0, 
                                  k: int = 10
                                 ) -> pd.DataFrame:
    """
    Trains an ExtraTreesRegressor model on the given dataframe and returns the k most important features.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe with predictor and target variables.
    target_col : str
        The name of the target variable.
    figsize : tuple, optional
        The size of the matplotlib figure to display the feature importances (default is (10, 15)).
    encoder : str, optional
        The type of encoder to use on the categorical variables. Can be either 'ohe' for OneHotEncoder or 'le' for LabelEncoder (default is None).
    ohe_cols : list, optional
        The names of the columns to apply OneHotEncoder to. If not specified, all categorical columns are encoded (default is None).
    le_cols : list, optional
        The names of the columns to apply LabelEncoder to. If not specified, all categorical columns are encoded (default is None).
    n_estimators : int, optional
        The number of trees in the forest (default is 100).
    criterion : str, optional
        The function to measure the quality of a split. Can be either 'mse' for mean squared error or 'mae' for mean absolute error (default is 'squared_error').
    max_features : float, optional
        The maximum number of features each tree is allowed to use (default is 1.0).
    k : int, optional
        The number of top features to return (default is 10).

    Returns:
    --------
    pd.DataFrame
        A dataframe with the k most important features and their corresponding importances.

    """
    # drop rows with na
    df = df.dropna(axis=1)

    # Separate the predictors and target variable
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Apply one-hot encoding if specified
    if ohe_cols:
        categorical_cols = list(set(ohe_cols) & set(X.columns))
        if categorical_cols:
            ohe = OneHotEncoder()
            ohe.fit(X[categorical_cols])
            encoded_cols = ohe.get_feature_names_out(categorical_cols)
            X_encoded = pd.concat([X.drop(categorical_cols, axis=1), 
                                   pd.DataFrame(ohe.transform(X[categorical_cols]).toarray(), 
                                                columns=encoded_cols)], axis=1)
            X = X_encoded
   # Apply encoding if specified
    elif encoder == "ohe":
    # One-hot encode categorical variables
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        if categorical_cols:
            ohe = OneHotEncoder()
            ohe.fit(X[categorical_cols])
            encoded_cols = ohe.get_feature_names_out(categorical_cols)
            X_encoded = pd.concat([X.drop(categorical_cols, axis=1), 
                                   pd.DataFrame(ohe.transform(X[categorical_cols]).toarray(), 
                                                columns=encoded_cols)], axis=1)
            X = X_encoded
    
    # Apply label encoding if specified
    if le_cols:
        categorical_cols = list(set(le_cols) & set(X.columns))
        if categorical_cols:
            le = LabelEncoder()
            X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))
    elif encoder == "le":
        # Label encode categorical variables
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        if categorical_cols:
            le = LabelEncoder()
            X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

    # Delete categorical vars not specified to encode
    cat_vars = X.select_dtypes(include=['category', 'object']).columns.tolist()
    if cat_vars: # if not empty
        # Remove columns with categorical variables
        categorical_cols_in_data = set(cat_vars).intersection(set(X.columns))
        X = X.drop(categorical_cols_in_data, axis=1)
    
    model = ExtraTreesRegressor(n_estimators = n_estimators, criterion =criterion, max_features = max_features, random_state=42) # max_features = 0.3
    model.fit(X,y)
    # use inbuilt class feature importances of tree based classitiers 
    #print(model.feature_importances_)

    # perform feature important using the extra_tree_forest.feature_importances_
    # plot graph of feature importances for better visualization
    
    k = k # k most important features
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(k).plot(kind='barh')
    plt.show()
    
    return feat_importances.nlargest(k).sort_values(ascending=False)
    
def remove_outliers_iqr(df: pd.DataFrame, cols: List[str], threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a pandas DataFrame using the interquartile range (IQR) method.
    
    Args:
        df: A pandas DataFrame containing the data.
        cols: A list of columns to remove outliers from.
        threshold: The number of IQRs beyond which a data point is considered an outlier.
                   Defaults to 1.5.
                   
    Returns:
        A pandas DataFrame with the outliers removed.
    """
    # Calculate the IQR for each specified column
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1
    
    # Determine the threshold values for each specified column
    thresholds = (Q1 - threshold * IQR, Q3 + threshold * IQR)
    
    # Identify and remove the outliers
    is_outlier = ((df[cols] < thresholds[0]) | (df[cols] > thresholds[1])).any(axis=1)
    df_clean = df.loc[~is_outlier, :]
    
    return df_clean

def scale_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Applies MinMaxScaler to the specified columns of a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to scale.
    columns : List[str]
        The list of column names to scale.

    Returns:
    --------
    pd.DataFrame
        The scaled pandas DataFrame.
    """

    # Create a copy of the DataFrame
    df_scaled = df.copy()

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Scale the specified columns
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])

    return df_scaled


def generate_train_test_sets(df_train: pd.DataFrame, 
                             df_test: pd.DataFrame, 
                             target: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
            Generate training and test sets.

            Args:
                df_train (pd.DataFrame): Training dataset.
                df_test (pd.DataFrame): Test dataset.
                target (str): Target variable name.

            Returns:
                X_train (pd.DataFrame): Training features.
                y_train (pd.Series): Training target variable.
                X_test (pd.DataFrame): Test features.
                y_test (pd.Series): Test target variable.
        """
        X_train = df_train.drop(columns=[target])
        y_train = df_train[target]

        X_test = df_test.drop(columns=[target])
        y_test = df_test[target]

        return X_train, y_train, X_test, y_test

