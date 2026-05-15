import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, KNNImputer
from collections import Counter
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import  RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


app = ctk.CTk()
app.geometry("640x480") 
app._set_appearance_mode("Dark") 
app.title("ML Project")

fram = ctk.CTkFrame(app, fg_color="black")
fram.pack(fill="both", expand=True)

def clear_frame():
    for widget in fram.winfo_children():
        widget.destroy()

current_data = None
selected_columns = []

def upload_dataset():
    global current_data,selected_columns
    file_path = filedialog.askopenfilename(title="Upload Dataset",filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            current_data = pd.read_csv(file_path)
            current_data = current_data.copy()
            selected_columns = []
            update_column_types()
            Page_zero()
        except Exception as e:
            print("Failed to load dataset:", e)

def update_column_types():
    global float_cols, object_cols
    float_cols = current_data.select_dtypes(include=['float64', 'int']).columns
    object_cols = current_data.select_dtypes(include=['object']).columns 
    
def page_base ():     
    clear_frame()
    ctk.CTkLabel(fram, text="Welcome to the Project", font=("arial", 15, "bold")).pack(padx=100, pady=10)

    ctk.CTkButton(fram, text="Upload Dataset", command=upload_dataset, corner_radius=30,hover_color="green").pack(pady=5)
                 
def Page_zero():
    clear_frame()
    ctk.CTkLabel(fram, text="Welcome to the Project", font=("arial", 15, "bold")).pack(padx=100, pady=10)
    bot_data_view = ctk.CTkButton(fram, text="1-View Dataset", command=show_dataset, corner_radius=30, hover_color="green")
    bot_data_view.pack(pady=10)
    bot1=ctk.CTkButton(fram, text="2-Data Visualization", command=Data_Visualization_be, corner_radius=30,hover_color="green")
    bot1.pack(pady=10)
    bot2=ctk.CTkButton(fram, text="3-Preprocessing", command=Preprocessing, corner_radius=30,hover_color="green")
    bot2.pack(pady=10)
    bot3=ctk.CTkButton(fram, text="4-Data Visualization", command=Data_Visualization_af, corner_radius=30,hover_color="green")
    bot3.pack(pady=10)
    bot4=ctk.CTkButton(fram, text="5-ML Models", command=ML_Models, corner_radius=30,hover_color="green")
    bot4.pack(pady=10)
    bot_back=ctk.CTkButton(fram, text="Back", command=page_base, corner_radius=30)
    bot_back.pack(pady=10)

def show_dataset():
    clear_frame()
    ctk.CTkLabel(fram, text="Dataset Preview", font=("arial", 15, "bold")).pack(pady=10)
    tree_frame = tk.Frame(fram)
    tree_frame.pack(fill="both", expand=True)

    tree_scroll_y = tk.Scrollbar(tree_frame)
    tree_scroll_y.pack(side="right", fill="y")
    tree_scroll_x = tk.Scrollbar(tree_frame, orient="horizontal")
    tree_scroll_x.pack(side="bottom", fill="x")

    tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
    tree.pack(fill="both", expand=True)
    tree_scroll_y.config(command=tree.yview)
    tree_scroll_x.config(command=tree.xview)

    tree["column"] = list(current_data.columns)
    tree["show"] = "headings"

    for col in current_data.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center")
    for index, row in current_data.head(100).iterrows():
        tree.insert("", "end", values=list(row))
        
    ctk.CTkButton(fram, text="Back", command=Page_zero, corner_radius=30).pack(pady=5)
           
def Data_Visualization_be():
    clear_frame()
    ctk.CTkLabel(fram, text=" Data Visualization Before Preprocessing ", font=("arial", 15, "bold")).pack(padx=100, pady=10)

    bot1=ctk.CTkButton(fram, text="Histogram", command=plot_histograms, corner_radius=30,hover_color="green")
    bot1.pack(pady=5)
    bot2=ctk.CTkButton(fram, text="Box Plot", command=plot_boxplots, corner_radius=30,hover_color="green")
    bot2.pack(pady=5)
    bot3=ctk.CTkButton(fram, text="Scatter Plot", command=plot_scatter_all, corner_radius=30,hover_color="green")
    bot3.pack(pady=5)
    bot4=ctk.CTkButton(fram, text="Heatmap", command=plot_heatmap, corner_radius=30,hover_color="green")
    bot4.pack(pady=5)
    bot5=ctk.CTkButton(fram, text="Bar Chart", command=plot_bar_chart, corner_radius=30,hover_color="green")
    bot5.pack(pady=5)
    bot6=ctk.CTkButton(fram, text="Pie Chart", command=plot_pie_chart, corner_radius=30,hover_color="green")
    bot6.pack(pady=5)
    bot_black=ctk.CTkButton(fram, text="Back", command=Page_zero, corner_radius=30)
    bot_black.pack(pady=10)

def init_selected_columns(n=9):
    global selected_columns, current_data
    if current_data is None:
        ctk.CTkLabel(fram, text="Error: No dataset selected").pack()
        return False
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        ctk.CTkLabel(fram, text="Error: No numeric columns available for plotting").pack()
        return False
    n = min(n, len(numeric_cols))
    selected_columns[:] = list(np.random.choice(numeric_cols, size=n, replace=False))
    return True

def plot_histograms():
    global selected_columns , current_data
    init_selected_columns()
    num_plots = len(selected_columns)
    rows = (num_plots // 3) 
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(selected_columns):
        axes[i].hist(current_data[col], bins=60, edgecolor="white")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_boxplots():
    global selected_columns , current_data
    num_plots = len(selected_columns)
    rows = (num_plots // 3) 
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(selected_columns):
        sns.boxplot(ax=axes[i], data=current_data, y=col, palette='Spectral')
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_bar_chart():
    global selected_columns , current_data
    num_plots = len(selected_columns)
    rows = (num_plots // 3) 
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(selected_columns):
        top = current_data[col].value_counts().head(7)
        top.plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[i])
        axes[i].set_title(f'Top 7 in {col}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_pie_chart():
    global selected_columns , current_data
    num_plots = len(selected_columns)
    rows = (num_plots // 3) 
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(selected_columns):
        top = current_data[col].value_counts().head(7)
        axes[i].pie(top, labels=top.index, autopct='%1.1f%%',explode=[0.05] * len(top), shadow=True,textprops={'fontsize': 10})
        axes[i].set_title(f'Top 7 in {col}')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_scatter_all():
    if current_data is None:
        ctk.CTkLabel(fram, text="Error: No dataset selected").pack()
        return
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        ctk.CTkLabel(fram, text="Error: Not enough numeric columns for scatter plots").pack()
        return
    n_pairs = min(6, len(numeric_cols) // 2)
    selected = list(np.random.choice(numeric_cols, size=n_pairs * 2, replace=False))
    pairs = [(selected[i], selected[i+1]) for i in range(0, len(selected), 2)]
    rows = (len(pairs) // 3)
    fig, axs = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axs = axs.flatten()
    for i, (x, y) in enumerate(pairs):
        sns.scatterplot(x=current_data[x], y=current_data[y], ax=axs[i])
        axs[i].set_title(f'{x} vs {y}')
        axs[i].set_xlabel(x)
        axs[i].set_ylabel(y)
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()

def plot_heatmap():
    if current_data is None:
        ctk.CTkLabel(fram, text="Error: No dataset selected").pack()
        return
    corr = current_data.select_dtypes(include='number').corr()
    if corr.empty:
        ctk.CTkLabel(fram, text="Error: No numeric data for heatmap").pack()
        return
    plt.figure(figsize=(min(2 + len(corr.columns), 20), min(2 + len(corr.columns), 20)))
    sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
    plt.tight_layout()
    plt.show()

################################################################################################################################

def Preprocessing ():
    clear_frame()
    label = ctk.CTkLabel(fram, text=" Preprocessing ", font=("arial", 15, "bold"))
    label.pack(padx=100,pady=10)
    
    bot1=ctk.CTkButton(fram, text=" Handle Missing Values ",width=150,height=25, command=handel_missing, corner_radius=30, hover_color="green")
    bot1.pack(pady=10)
    bot2=ctk.CTkButton(fram, text=" Remove Outliers ",width=150,height=25, command=remove_outlier, corner_radius=30, hover_color="green")
    bot2.pack(pady=10)
    bot3 = ctk.CTkButton(fram, text="Normalize Data", width=150, height=25, command=normalize_data, corner_radius=30, hover_color="green")
    bot3.pack(pady=10)
    bot4 = ctk.CTkButton(fram, text="Encode Data", width=150, height=25, command=encode_data, corner_radius=30, hover_color="green")
    bot4.pack(pady=10)
    bot_back=ctk.CTkButton(fram, text="Back", command=Page_zero, corner_radius=30)
    bot_back.pack(pady=20)
    
def handel_missing ():
    clear_frame()
    label = ctk.CTkLabel(fram, text=" Handle Missing Values ", font=("arial", 15, "bold"))
    label.pack(padx=100,pady=10)
    bot1=ctk.CTkButton(fram, text=" Drop Unnecessary Columns ",width=150,height=25, command=select_columns_to_drop, corner_radius=30, hover_color="green")
    bot1.pack(pady=10)    
    bot2=ctk.CTkButton(fram, text=" Drop Duplicate Data ",width=150,height=25, command=duplicate_data, corner_radius=30, hover_color="green")
    bot2.pack(pady=10)    
    bot3=ctk.CTkButton(fram, text=" Drop Nan Value ",width=150,height=25, command=drop_data, corner_radius=30, hover_color="green")
    bot3.pack(pady=10)
    bot4=ctk.CTkButton(fram, text=" Simple Imputer ",width=150,height=25, command=simple_imputer, corner_radius=30, hover_color="green")
    bot4.pack(pady=10)
    bot5=ctk.CTkButton(fram, text=" KNN Imputer ",width=150,height=25, command=KNN_imputer, corner_radius=30, hover_color="green")
    bot5.pack(pady=10)
    bot6=ctk.CTkButton(fram, text=" Iterative Imputer ",width=150,height=25, command=iterative_imputer, corner_radius=30, hover_color="green")
    bot6.pack(pady=10)        
    bot_back=ctk.CTkButton(fram, text="Back", command=Preprocessing, corner_radius=30)
    bot_back.pack(pady=20) 
    
def simple_imputer():
    global current_data
    update_column_types()
    current_data[float_cols] = current_data[float_cols].fillna(current_data[float_cols].mean())
    for col in object_cols:
        current_data[col] = current_data[col].fillna(current_data[col].mode()[0])
    ctk.CTkLabel(fram, text="Fill Missing Value (Simple Imputer) Successfully").pack(pady=10)

def KNN_imputer():
    global current_data
    numeric_cols = current_data.select_dtypes(include=['number'])
    knn_imputer = KNNImputer(n_neighbors=3)
    imputed_data = knn_imputer.fit_transform(numeric_cols)
    current_data[numeric_cols.columns] = imputed_data 
    ctk.CTkLabel(fram, text="Apply KNN Imputer Successfully").pack(pady=10)

def iterative_imputer():
    global current_data
    imputer = IterativeImputer(max_iter=10, random_state=42)
    imputed_data = imputer.fit_transform(current_data.select_dtypes(include=['number']))
    current_data[current_data.select_dtypes(include=['number']).columns] = imputed_data
    ctk.CTkLabel(fram, text="Apply Iterative Imputer Successfully").pack(pady=10)

def select_columns_to_drop():
    global current_data
    clear_frame()
    ctk.CTkLabel(fram, text="Select Columns to Drop", font=("arial", 15, "bold")).pack(pady=10)
    scroll_frame = ctk.CTkScrollableFrame(fram, width=400, height=300)
    scroll_frame.pack(pady=10)
    column_vars = {} 
    for col in current_data.columns:
        var = tk.BooleanVar()
        chk = ctk.CTkCheckBox(scroll_frame, text=col, variable=var)
        chk.pack(anchor="w")
        column_vars[col] = var

    def confirm_drops():
        global current_data
        to_drop = [col for col, var in column_vars.items() if var.get()]
        if to_drop:
            current_data.drop(columns=to_drop, inplace=True)
            ctk.CTkLabel(fram,text=f"Dropped Dropped columns: {', '.join(to_drop)}")
            handel_missing()
        else:
            ctk.CTkLabel(fram,text="No Selection No columns were selected for dropping")

    ctk.CTkButton(fram, text="Confirm Drop", command=confirm_drops, corner_radius=30,hover_color="green").pack(pady=10)
    ctk.CTkButton(fram, text="Back", command=handel_missing, corner_radius=30).pack(pady=10)

def duplicate_data ():
    global current_data
    current_data.drop_duplicates()
    label_message = ctk.CTkLabel(fram, text="Drop Duplicate Data Successfully")
    label_message.pack(pady=10)      
    
def drop_data():
    global current_data
    current_data.dropna(inplace=True)
    label_message = ctk.CTkLabel(fram, text="Dropping Nan Value Successfully")
    label_message.pack(pady=10)  
     
def remove_outlier ():
    clear_frame()
    label = ctk.CTkLabel(fram, text=" Remove Outlier ", font=("arial", 15, "bold"))
    label.pack(padx=100,pady=10)

    bot1=ctk.CTkButton(fram,text=" IQR ",width=150,height=25,command=IQR,corner_radius=30,hover_color="green")
    bot1.pack(padx=100,pady=10)
    bot2=ctk.CTkButton(fram,text=" Standard Distribution ",width=150,height=25,command=Standard_Distribution,corner_radius=30,hover_color="green")
    bot2.pack(padx=100,pady=10)
    bot3=ctk.CTkButton(fram,text="  Z-Score ",width=150,height=25,command=Z_Score,corner_radius=30,hover_color="green")
    bot3.pack(padx=100,pady=10)
    bot4=ctk.CTkButton(fram,text=" Modified Z-Score ",width=150,height=25,command=Modified_Z_Score,corner_radius=30,hover_color="green")
    bot4.pack(padx=100,pady=10)
    bot_back=ctk.CTkButton(fram, text="Back", command=Preprocessing, corner_radius=30)
    bot_back.pack(pady=10)

def IQR():
    global current_data
    numeric_cols = current_data.select_dtypes(include=['number'])
    total_outliers = 0
    
    for col in numeric_cols:
        Q1 = current_data[col].quantile(0.25)
        Q3 = current_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        is_outlier = (current_data[col] < lower_bound) | (current_data[col] > upper_bound)
        outlier_count = is_outlier.sum()
        total_outliers += outlier_count
        current_data = current_data[~is_outlier]
    print(f"Remove Outlier IQR \n Number outliers removing : {total_outliers}") 
    label_message = ctk.CTkLabel(fram, text=f"Remove Outlier IQR \n Number outliers removing : {total_outliers}")
    label_message.pack(pady=10)
    
def Standard_Distribution():
    global current_data
    outlier_indices = []
    features = current_data.select_dtypes(include=['float64', 'int']).columns

    for column in features:
        data_mean = current_data[column].mean()
        data_std = current_data[column].std()
        cut_off = data_std * 3
        outliers = current_data[(current_data[column] < data_mean - cut_off) | (current_data[column] > data_mean + cut_off)].index
        outlier_indices.extend(outliers)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = [k for k, v in outlier_indices.items() if v > 1]

    current_data.drop(index=multiple_outliers, inplace=True)
    current_data.reset_index(drop=True, inplace=True)
    print("Total number of outliers (Standard Distribution):", len(multiple_outliers))
    label_message = ctk.CTkLabel(fram, text=f"Remove Outlier Standard Distribution \n Number outliers removing : {len(multiple_outliers)}")
    label_message.pack(pady=10)
    
def Z_Score():
    global current_data
    outlier_list = []
    features = current_data.select_dtypes(include=['float64', 'int']).columns
    
    for column in features:
        data_mean = current_data[column].mean()
        data_std = current_data[column].std()
        threshold = 3
        z_score = abs((current_data[column] - data_mean) / data_std)

        outlier_list_column = current_data[z_score > threshold].index
        outlier_list.extend(outlier_list_column)
        
    outlier_list = Counter(outlier_list)
    multiple_outliers = [k for k, v in outlier_list.items() if v > 1]
    current_data.drop(index=multiple_outliers, inplace=True)
    current_data.reset_index(drop=True, inplace=True)
    print("Total number of outliers (Z-Score):", len(multiple_outliers))
    label_message = ctk.CTkLabel(fram, text=f"Remove Outlier Z Score \n Number outliers removing : {len(multiple_outliers)}")
    label_message.pack(pady=10)
     

def Modified_Z_Score():
    global current_data
    outlier_list = []
    features = current_data.select_dtypes(include=['float64', 'int']).columns
    for column in features:
        data_median = current_data[column].median()
        MAD = median_abs_deviation(current_data[column], nan_policy='omit')
        threshold = 3.5 
        mod_z_score = abs(0.6745 * (current_data[column] - data_median) / MAD)
        outlier_list_column = current_data[mod_z_score > threshold].index
        outlier_list.extend(outlier_list_column)

    outlier_list = Counter(outlier_list)
    multiple_outliers = [k for k, v in outlier_list.items() if v > 1]

    current_data.drop(index=multiple_outliers, inplace=True)
    current_data.reset_index(drop=True, inplace=True)
    print("Total number of outliers (Modified Z-Score):", len(multiple_outliers))
    label_message = ctk.CTkLabel(fram, text=f"Remove Outlier Modified Z-Score \n Number outliers removing : {len(multiple_outliers)}")
    label_message.pack(pady=10)

def normalize_data():
    clear_frame()
    ctk.CTkLabel(fram, text="Normalization", font=("arial", 15, "bold")).pack(pady=10)

    def min_max_normalize():
        global current_data
        numeric_cols = current_data.select_dtypes(include=['float64', 'int']).columns
        current_data[numeric_cols] = (current_data[numeric_cols] - current_data[numeric_cols].min()) / (current_data[numeric_cols].max() - current_data[numeric_cols].min())
        current_data.to_csv("normalized_data_minmax.csv", index=False)
        ctk.CTkLabel(fram, text="Min-Max Normalization Applied").pack(pady=10)

    def Standard_Scalernormalize():
        global current_data
        numeric_cols = current_data.select_dtypes(include=['float64', 'int']).columns
        scaler = StandardScaler()
        current_data[numeric_cols] = scaler.fit_transform(current_data[numeric_cols])
        ctk.CTkLabel(fram, text="Standard Scaler Normalization Applied").pack(pady=10)
    bot1=ctk.CTkButton(fram, text="Min-Max Normalization", command=min_max_normalize, corner_radius=30, hover_color="green")
    bot1.pack(pady=10)
    bot2=ctk.CTkButton(fram, text="Standard Scaler Normalization", command=Standard_Scalernormalize, corner_radius=30, hover_color="green")
    bot2.pack(pady=10)
    bot_black=ctk.CTkButton(fram, text="Back", command=Preprocessing, corner_radius=30)
    bot_black.pack(pady=20)

def encode_data():
    clear_frame()
    bot1=ctk.CTkLabel(fram, text="Encode Categorical Features", font=("arial", 15, "bold"))
    bot1.pack(pady=10)
    bot2=ctk.CTkButton(fram, text="Label Encoding", command=label_encode, corner_radius=30, hover_color="green")
    bot2.pack(pady=10)
    bot3=ctk.CTkButton(fram, text="One-Hot Encoding", command=one_hot_encode, corner_radius=30, hover_color="green")
    bot3.pack(pady=10)
    bot_black=ctk.CTkButton(fram, text="Back", command=Preprocessing, corner_radius=30)
    bot_black.pack(pady=20)

def label_encode():
    global current_data
    update_column_types()
    print("Label Encoding applied to columns:", list(object_cols))
    le = LabelEncoder()
    for col in object_cols:
        try:
            current_data[col] = le.fit_transform(current_data[col].astype(str))
        except:
            continue
    current_data.to_csv("encoded_label_data.csv", index=False)
    ctk.CTkLabel(fram, text=f"Label Encoding Applied \n Columns = {list(object_cols)}").pack(pady=10)

def one_hot_encode():
    global current_data
    update_column_types()
    print("One-Hot Encoding applied to columns:", list(object_cols))
    try:
        current_data = pd.get_dummies(current_data, columns=object_cols)
        current_data.to_csv("encoded_onehot_data.csv", index=False)
        ctk.CTkLabel(fram, text=f"One-Hot Encoding Applied \n Columns = {list(object_cols)}").pack(pady=10)
    except Exception as e:
        ctk.CTkLabel(fram, text=f"Encoding Error: {e}", text_color="red").pack(pady=10)

###################################################################################################################################

def Data_Visualization_af ():
    clear_frame()
    label = ctk.CTkLabel(fram, text=" Data Visualization After Perprocessing ", font=("arial", 15, "bold"))
    label.pack(padx=100,pady=10)

    bot1=ctk.CTkButton(fram, text=" Histogram ",command=plot_histograms_af, corner_radius=30, hover_color="green")
    bot1.pack(pady=10)
    bot2=ctk.CTkButton(fram, text=" Box plot ",command=plot_boxplots_af, corner_radius=30, hover_color="green")
    bot2.pack(pady=10)
    bot3=ctk.CTkButton(fram, text=" Scatter plot ",command=plot_scatter_all_af, corner_radius=30, hover_color="green")
    bot3.pack(pady=10)
    bot4=ctk.CTkButton(fram, text=" Heatmap ",command=plot_heatmap_af, corner_radius=30, hover_color="green")
    bot4.pack(pady=10)
    bot5=ctk.CTkButton(fram, text=" Bar chart ",command=plot_bar_chart_af, corner_radius=30, hover_color="green")
    bot5.pack(pady=10)
    bot6=ctk.CTkButton(fram, text=" Pie chart ",command=plot_pie_chart_af, corner_radius=30, hover_color="green")
    bot6.pack(pady=10)
    bot_back=ctk.CTkButton(fram, text="Back",command=Page_zero, corner_radius=30)
    bot_back.pack(pady=10)
    
def plot_histograms_af():
    global selected_columns , current_data
    init_selected_columns()
    num_plots = len(selected_columns)
    rows = (num_plots // 3) 
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(selected_columns):
        axes[i].hist(current_data[col], bins=60, edgecolor="white")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_boxplots_af():
    global selected_columns , current_data
    num_plots = len(selected_columns)
    rows = (num_plots // 3) 
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(selected_columns):
        sns.boxplot(ax=axes[i], data=current_data, y=col, palette='Spectral')
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_bar_chart_af():
    global selected_columns , current_data
    num_plots = len(selected_columns)
    rows = (num_plots // 3) 
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(selected_columns):
        top = current_data[col].value_counts().head(7)
        top.plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[i])
        axes[i].set_title(f'Top 7 in {col}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_pie_chart_af():
    global selected_columns , current_data
    num_plots = len(selected_columns)
    rows = (num_plots // 3) 
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i, col in enumerate(selected_columns):
        top = current_data[col].value_counts().head(7)
        axes[i].pie(top, labels=top.index, autopct='%1.1f%%',explode=[0.05] * len(top), shadow=True,textprops={'fontsize': 10})
        axes[i].set_title(f'Top 7 in {col}')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_scatter_all_af():
    if current_data is None:
        print("No dataset selected.")
        return
    numeric_cols = current_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        ctk.CTkLabel(fram, text="Error: Not enough numeric columns for scatter plots").pack()
        return

    n_pairs = min(6, len(numeric_cols) // 2)
    selected = list(np.random.choice(numeric_cols, size=n_pairs * 2, replace=False))
    pairs = [(selected[i], selected[i+1]) for i in range(0, len(selected), 2)]

    rows = (len(pairs) // 3)
    fig, axs = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    axs = axs.flatten()
    for i, (x, y) in enumerate(pairs):
        sns.scatterplot(x=current_data[x], y=current_data[y], ax=axs[i])
        axs[i].set_title(f'{x} vs {y}')
        axs[i].set_xlabel(x)
        axs[i].set_ylabel(y)
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()

def plot_heatmap_af():
    if current_data is None:
        print("No dataset selected")
        return
    corr = current_data.select_dtypes(include='number').corr()
    if corr.empty:
        ctk.CTkLabel(fram, text="Error: No numeric data for heatmap").pack()
        return
    
    plt.figure(figsize=(min(2 + len(corr.columns), 20), min(2 + len(corr.columns), 20)))
    sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
    plt.tight_layout()
    plt.show()

#####################################################################################################################################

def ML_Models ():
    clear_frame()
    label = ctk.CTkLabel(fram, text="Machine Learning Models", font=("arial", 15, "bold"))
    label.pack(pady=30)

    bot1=ctk.CTkButton(fram, text="Select Target Column", command=select_target_column, corner_radius=30, hover_color="green")
    bot1.pack(pady=10)
    bot2=ctk.CTkButton(fram, text="Unsupervised Models", command=show_unsupervised, corner_radius=30, hover_color="green")
    bot2.pack(pady=10)
    bot_back=ctk.CTkButton(fram, text="Back", command=Page_zero, corner_radius=30)
    bot_back.pack(pady=20)

selected_target = None 

def select_target_column():
    global selected_target, current_data
    clear_frame()
    label = ctk.CTkLabel(fram, text="Select Target Column", font=("arial", 15, "bold"))
    label.pack(pady=10)
    combo = ctk.CTkComboBox(fram, values=list(current_data.columns))
    combo.pack(pady=10)

    def confirm_selection():
        global selected_target, x, y
        selected_target = combo.get()
        x = current_data.drop(columns=[selected_target])
        y = current_data[selected_target]
        ctk.CTkLabel(fram, text=f"Target SetTarget column set to: {selected_target}").pack(pady=10)
        show_supervised()

    bot1=ctk.CTkButton(fram, text="Confirm", command=confirm_selection,corner_radius=30,)
    bot1.pack(pady=10)
    bot_back=ctk.CTkButton(fram, text="Back", command=Page_zero,corner_radius=30)
    bot_back.pack(pady=10)

def show_supervised():
        clear_frame()
        label = ctk.CTkLabel(fram, text="Supervised Models", font=("arial", 15, "bold"))
        label.pack(pady=20)

        bot1=ctk.CTkButton(fram, text="Regression Models", command=show_regression, corner_radius=30, hover_color="green")
        bot1.pack(pady=10)
        bot2=ctk.CTkButton(fram, text="Classification Models", command=show_classification, corner_radius=30, hover_color="green")
        bot2.pack(pady=10)
        bot_back=ctk.CTkButton(fram, text="Back", command=ML_Models, corner_radius=30)
        bot_back.pack(pady=20)
        
def show_regression():
    clear_frame()
    label = ctk.CTkLabel(fram, text="Regression Models", font=("arial", 15, "bold"))
    label.pack(pady=20)
        
    def Linear_Regression ():
        global x, y
        model = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(mse)
        r2 = r2_score(y_test, y_pred)
        print(r2)
        ctk.CTkLabel(fram, text=f"Linear Regression model Applied \n mean squared error = {mse} , \n R2 Score = {r2}").pack(pady=10)
        
    def Decision_Tree ():
        global x, y
        model = DecisionTreeRegressor(criterion="squared_error",max_depth=3)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        plt.figure(figsize=(14, 8))
        plot_tree(model, feature_names=x.columns.tolist(), filled=True, fontsize=10)
        plt.title("Decision Tree Regressor")
        plt.show()
        print("MSE:", mse)
        print("R2_Score:", r2)
        ctk.CTkLabel(fram, text=f"Decision Tree model Applied \n mean squared error = {mse} , \n R2_Score = {r2}").pack(pady=10)
       
    def Random_Forest ():
        global x, y
        model = RandomForestRegressor()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(mse)
        r2 = r2_score(y_test, y_pred)
        print(r2)
        ctk.CTkLabel(fram, text=f"Random Forest model Applied \n mean squared error = {mse} , \n R2 Score = {r2}").pack(pady=10)

    def K_Fold_Regression():
        global x, y
        model = LinearRegression()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores = cross_val_score(model, x, y, cv=kf, scoring='r2')
        avg_r2 = np.mean(r2_scores)
        result = f"K-Fold (5 splits)\nR² Scores: {r2_scores.round(4)}\nAverage R²: {avg_r2:.4f}"
        print(result)
        ctk.CTkLabel(fram, text=result).pack(pady=10)        

    bot1=ctk.CTkButton(fram, text="Linear Regression", command=Linear_Regression, corner_radius=30, hover_color="green")
    bot1.pack(pady=10)
    bot2=ctk.CTkButton(fram, text="Decision Tree", command=Decision_Tree, corner_radius=30, hover_color="green")
    bot2.pack(pady=10)
    bot3=ctk.CTkButton(fram, text="Random Forest", command=Random_Forest, corner_radius=30, hover_color="green")
    bot3.pack(pady=10)
    bot4=ctk.CTkButton(fram, text="K-Fold Regression", command=K_Fold_Regression, corner_radius=30, hover_color="green")
    bot4.pack(pady=10)
    bot_back=ctk.CTkButton(fram, text="Back", command=show_supervised, corner_radius=30, hover_color="green")
    bot_back.pack(pady=20)

def show_classification():
    clear_frame()
    label = ctk.CTkLabel(fram, text="Classification Models", font=("arial", 15, "bold"))
    label.pack(pady=20)
        
    def Logistic_Regression ():
        global x, y,current_data
        model = LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print("Report accuracy \n ",report)
        print("Accuracy = ",acc)
        ctk.CTkLabel(fram, text="Logistic Regression model Applied ").pack(pady=10)
        ctk.CTkLabel(fram, text=f"Report accuracy =\n {report}").pack(pady=10)
        ctk.CTkLabel(fram, text=f"Accuracy = {acc}").pack(pady=10)        
        
    def SVM ():
        global x, y,current_data
        model = SVC()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print("Report accuracy \n",report)
        print("Accuracy = ",acc)
        ctk.CTkLabel(fram, text="SVM model Applied ").pack(pady=10)
        ctk.CTkLabel(fram, text=f"Report accuracy =\n {report}").pack(pady=10)
        ctk.CTkLabel(fram, text=f"Accuracy = {acc}").pack(pady=10)        
        
    def K_NN ():
        global x, y,current_data
        model = KNeighborsClassifier()
        X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print("Report accuracy \n",report)
        print("Accuracy = ",acc)
        ctk.CTkLabel(fram, text="K-NN model Applied ").pack(pady=10)
        ctk.CTkLabel(fram, text=f"Report accuracy =\n {report}").pack(pady=10)
        ctk.CTkLabel(fram, text=f"Accuracy = {acc}").pack(pady=10)
        error_rate = []
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 50),error_rate,color='black',markerfacecolor='green', markersize=10)
        plt.title('Error Rate vs. K-value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        plt.show()        
        
    def Naive_Bayes ():
        global x, y,current_data
        model = GaussianNB()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print("Report accuracy \n",report)
        print("Accuracy = ",acc)
        ctk.CTkLabel(fram, text="Naive Bayes model Applied ").pack(pady=10)
        ctk.CTkLabel(fram, text=f"Report accuracy =\n {report}").pack(pady=10)
        ctk.CTkLabel(fram, text=f"Accuracy = {acc}").pack(pady=10)
        
    def K_Fold_Classification():
        global x, y
        model = LogisticRegression()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        acc_scores = cross_val_score(model, x, y, cv=kf, scoring='accuracy')
        avg_acc = np.mean(acc_scores)
        result = f"K-Fold (5 splits)\nAccuracy Scores: {acc_scores.round(4)}\n Average Accuracy: {avg_acc:.4f}"
        print(result)
        ctk.CTkLabel(fram, text=result).pack(pady=10)        
                  
    bot1=ctk.CTkButton(fram, text="Logistic Regression", command=Logistic_Regression, corner_radius=30, hover_color="green")
    bot1.pack(pady=10)
    bot2=ctk.CTkButton(fram, text="SVM", command=SVM, corner_radius=30, hover_color="green")
    bot2.pack(pady=10)
    bot3=ctk.CTkButton(fram, text="K-NN", command=K_NN, corner_radius=30, hover_color="green")
    bot3.pack(pady=10)
    bot4=ctk.CTkButton(fram, text="Naive Bayes", command=Naive_Bayes, corner_radius=30, hover_color="green")
    bot4.pack(pady=10)
    bot5=ctk.CTkButton(fram, text="K-Fold Classification", command=K_Fold_Classification, corner_radius=30, hover_color="green")
    bot5.pack(pady=10)
    bot_back=ctk.CTkButton(fram, text="Back", command=show_supervised, corner_radius=30)
    bot_back.pack(pady=20)

def show_unsupervised():
    clear_frame()
    label = ctk.CTkLabel(fram, text="Unsupervised Models", font=("arial", 15, "bold"))
    label.pack(pady=20)
        
    def K_means():
        global current_data
        numeric_data = current_data.select_dtypes(include=np.number)
        if numeric_data.empty:
            ctk.CTkLabel(fram, text="Error No numeric columns available for clustering").pack(pady=10)
            return
        model = KMeans(n_clusters=3, random_state=42)
        labels = model.fit_predict(numeric_data)
        print(f"K-Means Clustering Complete\nClusters: {len(set(labels))}")
        ctk.CTkLabel(fram, text=f"K-Means Clustering Complete\nClusters: {len(set(labels))}").pack(pady=10)

    def UN_DBSCAN():
        global current_data
        numeric_data = current_data.select_dtypes(include=np.number)
        if numeric_data.empty:
            ctk.CTkLabel(fram, text="Error No numeric columns available for clustering").pack(pady=10)
            return
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(numeric_data)
        unique_labels = set(labels)
        num_clusters = len([label for label in unique_labels if label != -1])
        num_outliers = list(labels).count(-1)
        result_text = f"DBSCAN Clustering Complete\nClusters: {num_clusters}\nOutliers: {num_outliers}"
        print(result_text)
        ctk.CTkLabel(fram, text=result_text).pack(pady=10)
               
    bot1=ctk.CTkButton(fram, text="K-Means", command=K_means, corner_radius=30, hover_color="green")
    bot1.pack(pady=10)
    bot2=ctk.CTkButton(fram, text="DBSCAN", command=UN_DBSCAN, corner_radius=30, hover_color="green")
    bot2.pack(pady=10)
    bot_back=ctk.CTkButton(fram, text="Back", command=ML_Models, corner_radius=30)
    bot_back.pack(pady=20)
              
page_base()
app.mainloop()

