# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:26:45 2023

@author: sunle
"""

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
from ast import literal_eval
from scipy import stats
from sklearn.metrics import confusion_matrix
st.set_page_config(layout="wide")

mypath = "C:\\Users\\sunle\\covid_case_study\\"

filtered = pd.read_csv(mypath + "filtered.csv").drop("Unnamed: 0", axis = 1)
filtered[['level', "traits", "num_traits", "champions", 'star', "unit_cost"]] = filtered[['level', "traits", "num_traits", "champions", 'star', "unit_cost"]].apply(lambda x: x.astype("category"))
filtered["win"] = filtered.win.str.title()

cost_dict = {
    "Ahri":2,
    "Annie":2,
    "Ashe":3,
    "AurelionSol":5,
    "Blitzcrank":2,
    "Caitlyn":1,
    "ChoGath":4,
    "Darius":2,
    "Ekko":5,
    "Ezreal":3,
    "Fiora":1,
    "Fizz":4,
    "Gangplank":5,
    "Graves":1,
    "Irelia":4,
    "JarvanIV":1,
    "Jayce":3,
    "Jhin":4,
    "Jinx":4,
    "KaiSa":2,
    "Karma":3,
    "Kassadin":3,
    "Kayle":4,
    "KhaZix":1,
    "Leona":1,
    "Lucian":2,
    "Lulu":5,
    "Lux":3,
    "Malphite":1,
    "MasterYi":3,
    "MissFortune":5,
    "Mordekaiser":2,
    "Neeko":3,
    "Poppy":1,
    "Rakan":2,
    "Rumble":3,
    "Shaco":3,
    "Shen":2,
    "Sona":2,
    "Soraka":4,
    "Syndra":3,
    "Thresh":5,
    "TwistedFate":1,
    "VelKoz":4,
    "Vi":3,
    "WuKong":4,
    "Xayah":1,
    "Xerath":5,
    "XinZhao":2,
    "Yasuo":2,
    "Ziggs":1,
    "Zoe":1 
}


players_champs = filtered.groupby(["playerID", "ingameDuration", "win"]).agg({"comb_cost":np.sum, 'champions':list}).reset_index()

players_traits = filtered.copy()
players_traits["win"] = np.where(players_traits["Ranked"] <= 4, 1, 0)
players_traits = players_traits.groupby(["champions","comb_cost", "star"]).agg({"ingameDuration":np.mean, "win":np.mean}).reset_index()

def row_to_dict(row):
    return {'champions': row['champions'], 'star': row['star'], 'num_traits': row['num_traits']}

# Apply the function to each row to create the 'combined' column
# players['combined'] = tft.apply(row_to_dict, axis=1)

traits_dict = {
    "Ahri":["StarGuardian", "Set3_Sorcerer"],
    "Annie":["MechPilot","Set3_Sorcerer"],
    "Ashe":["Set3_Celestial","Sniper"],
    "AurelionSol":["Rebel","Starship"],
    "Blitzcrank":["Chrono","Set3_Brawler"],
    "Caitlyn":["Chrono","Sniper"],
    "ChoGath":["Set3_Void","Set3_Brawler"],
    "Darius":["SpacePirate","ManaReaver"],
    "Ekko":["Infiltrator","Cybernetic"],
    "Ezreal":["Chrono","Blaster"],
    "Fiora":["Cybernetic","Set3_Blademaster"],
    "Fizz":["MechPilot","Infiltrator"],
    "Gangplank":["SpacePirate","Demolitionist","Mercenary"],
    "Graves":["SpcaePirate","Blaster"],
    "Irelia":["Cybernetic","Set3_Blademaster","ManaReaver"],
    "JarvanIV":["DarkStar","Protector"],
    "Jayce":["SpcaePirate","Vanguard"],
    "Jhin":["DarkStar","Sniper"],
    "Jinx":["Rebel","Blaster"],
    "KaiSa":["Valkyrie","Infiltrator"],
    "Karma":["DarkStar","Set3_Mystic"],
    "Kassadin":["Set3_Celestial","ManaReaver"],
    "Kayle":["Valkyrie","Set3_Blademaster"],
    "KhaZix":["Set3_Void","Infiltrator"],
    "Leona":["Cybernetic","Vanguard"],
    "Lucian":["Cybernetic","Blaster"],
    "Lulu":["Set3_Celestial","Set3_Mystic"],
    "Lux":["DarkStar","Set3_Sorcerer"],
    "Malphite":["Rebel","Set3_Brawler"],
    "MasterYi":["Rebel","Set3_Blademaster"],
    "MissFortune":["Valkyrie","Mercenary","Blaster"],
    "Mordekaiser":["DarkStar","Vanguard"],
    "Neeko":["StarGuardian","Protector"],
    "Poppy":["StarGuardian","Vanguard"],
    "Rakan":["Set3_Celestial","Protector"],
    "Rumble":["MechPilot","Demolitionist"],
    "Shaco":["DarkStar","Infiltrator"],
    "Shen":["Chrono","Set3_Blademaster"],
    "Sona":["Rebel","Set3_Mystic"],
    "Soraka":["StarGuardian","Set3_Mystic"],
    "Syndra":["StarGuardian","Set3_Sorcerer"],
    "Thresh":["Chrono","ManaReaver"],
    "TwistedFate":["Chrono","Set3_Sorcerer"],
    "VelKoz":["Set3_Void","Sorcerer"],
    "Vi":["Cybernetic","Set3_Brawler"],
    "WuKong":["Chrono","Vanguard"],
    "Xayah":["Set3_Celestial","Set3_Blademaster"],
    "Xerath":["DarkStar","Set3_Sorcerer"],
    "XinZhao":["Set3_Celestial","Protector"],
    "Yasuo":["Rebel","Set3_Blademaster"],
    "Ziggs":["Rebel","Demolitionist"],
    "Zoe":["StarGuardian","Set3_Sorcerer"] 
}

champion_traits_list = [(champion, trait) for champion, traits in traits_dict.items() for trait in traits]


def fit_distribution(data):
    # estimate parameters
    mu = np.mean(data)
    sigma = np.std(data)
    # fit distribution
    dist = stats.norm(mu, sigma)
    return dist

def classifier(df, col_list, col):
    true_df = df[df[col] == "Win"][col_list].copy()
    false_df = df[df[col] == "Wose"][col_list].copy()
    prior = df[col].value_counts(normalize=True)
    true_copy = df[col_list].copy()
    false_copy = df[col_list].copy()
    
    for column in col_list:
        if (df[column].dtype == "category") == True:
            true_copy[column] = df[column].replace(dict(true_df[column].value_counts(normalize = True)))
            false_copy[column] = df[column].replace(dict(false_df[column].value_counts(normalize = True)))
        else:
            true_copy[column] = fit_distribution(true_df[column]).pdf(df[column])
            false_copy[column] = fit_distribution(false_df[column]).pdf(df[column])
            
    true = true_copy.prod(axis = 1, skipna = True) * prior["Win"]
    false = false_copy.prod(axis = 1, skipna = True) * prior["Lose"]
    
    result = pd.DataFrame(data = {"Given": df[col], "Pred": np.where(true > false, "Win", "Lose")})

    accuracy = np.mean(result.Given == result.Pred)
    
    return result, accuracy

def traits_map(k,v):
    return [(k,v[i]) for i in range(len(v))]

traits_list = [traits_map(k,v) for k,v in traits_dict.items()]

names = {k:v for k,v in zip(filtered.columns, ["Player ID", "Game ID", "Level", "Ranked", "In-game Duration", "Traits", "Number of Traits", "Champions", "Items", "Star", "Unit Cost", "Total Cost", "Win"])}


with st.sidebar: 
	selected = option_menu(
		menu_title = 'Navigation',
		options = ['Intro + Background','Data Cleaning','Exploratory Analysis','Data Analysis', 'Interactive Naive Bayes', 'Conclusion'],
		menu_icon = 'arrow-down-right-circle-fill',
		icons = ['circle','triangle','book', 'bar-chart', 'boxes', 'square'],
		default_index = 0,
		)
    
if selected=='Intro + Background':
    st.title("Aim of Study: ")
    st.header("TFT (Teamfight Tactics) is a popular auto-battler game based off DOTA's auto chess mod.")
    st.subheader("The aim of this study is to serve as a predecessor for using machine learning to accurately predict win & loss in ranked games and to find the best combinations of champions and traits at the start and throughout every major update (known as seasons, with a 'season x.5' gameplay shift in the middle of each season).")	
    st.subheader("In this study, I will first test sample data from season 3 using the naive bayes model, which is commonly used in classification with text and other elements. The model will analyse each row and return if they have won or not. This is useful as we can use this to find the most useful combination, and a formal launch of this service will be able to provide players with up-to-date information on the current season.")
    
    st.title("Background: ")
    st.header("Mechanics of the game: ")
    st.subheader("TFT is an auto-battler chess game, meaning that the player does not have to engage in the battle itself, and that they only have to organize thhe pieces that are playing each round. The game consists of eight players, and the game progresses in stages and rounds. Each stage, in order, consists of three combat rounds, a carousel to pick items from, and two more player combat rounds and one computer combat round.")
    
    st.header("Variables and their roles: ")
    st.subheader("The *level* column shows the level of a player. The higher the level, the more pieces the player can have and the pieces one gets from the shop will be stronger.")
    st.subheader("The *Ranked* column shows the rank of the player. In a normal ranked match, places 1-4 will still receive bonuses, therefore they are considered wins (& vice versa). ")
    st.subheader("The *ingameDuration* column is the time the player was involved in the game, measured in seconds")
    st.subheader("The *traits* column shows the traits the player currently has activated, and *num_traits* shows the number of pieces with that trait active.")
    st.subheader("The *star* column shows the star of the champion. The higher the star, the stronger the champion is in his *unit_cost* range. *Comb_cost* is the cost required to achieve the champion listed.")
    
if selected=='Data Cleaning':
    st.title("The Cleaning Process: ")	
    st.code("""
tft = pd.read_csv('tft.csv')
tft = tft.reset_index().rename(columns = {'index':'playerID'})

def comb_df(row):
    s = row["combination"]
    d = literal_eval(s)
    df = pd.DataFrame.from_dict(d, orient = "index").reset_index().rename(columns = {0:"num_traits", "index":"traits"})
    df["playerID"] = row["playerID"]
    df["gameId"] = row["gameId"]
    df["level"] = row["level"]
    df["Ranked"] = row["Ranked"]
    df["ingameDuration"] = row["ingameDuration"]
    return df
    
comb_df(tft.loc[0])

combination = tft[["playerID", "gameId", "combination", "level", "Ranked", "ingameDuration"]].copy()
combination.explode("combination")

def comb(s):
    d = literal_eval(s)
    return [[k,v] for k, v in d.items()]
combination["comb_list"] = combination["combination"].apply(comb)

comb_expl = combination.explode("comb_list")

final_comb_df = pd.concat([comb_df(row) for i, row in combination.iterrows()])
final_comb_df.to_csv("combination.csv", index = False)

traits_comb_cat = pd.read_csv("combination.csv")

champions = tft[["playerID", "gameId", "champion", "level", "Ranked", "ingameDuration"]].copy()
champions.explode("champion")

champions["champ_list"] = champions["champion"].apply(comb)
champions = champions.explode("champ_list")

def champs_df(row):
    s = row["champion"]
    d = literal_eval(s)
    df = pd.DataFrame.from_dict(d, orient = "index").reset_index().rename(columns = {0:"num_traits", "index":"champions"})
    df["playerID"] = row["playerID"]
    df["gameId"] = row["gameId"]
    return df

final_champs_df = pd.concat([champs_df(row) for i, row in champions.iterrows()]) 
            """)
    st.title("Sampling from merged DF")
    st.code("""
merged = traits_comb_cat.merge(final_champs_df, on=["playerID", "gameId"])
merged = merged.reindex(columns=['playerID', 'gameId', "level", "Ranked", "ingameDuration", 'traits', 'num_traits', 'champions', 'items', 'star'])
merged.to_csv("merged.csv", index = False)

test = pd.read_csv("merged.csv")

player_ids = test["playerID"].unique()
sample_ids = pd.Series(player_ids).sample(1000)
sample1_df = test[test["playerID"].isin(sample_ids)].copy()
sample1_df.to_csv("sample_main.csv")
            """)
            
    st.title("Adding the Combination Cost Column")
    st.code("""
cost_dict = {
    "Ahri":2,
    "Annie":2,
    "Ashe":3,
    "AurelionSol":5,
    "Blitzcrank":2,
    "Caitlyn":1,
    "ChoGath":4,
    "Darius":2,
    "Ekko":5,
    "Ezreal":3,
    "Fiora":1,
    "Fizz":4,
    "Gangplank":5,
    "Graves":1,
    "Irelia":4,
    "JarvanIV":1,
    "Jayce":3,
    "Jhin":4,
    "Jinx":4,
    "KaiSa":2,
    "Karma":3,
    "Kassadin":3,
    "Kayle":4,
    "KhaZix":1,
    "Leona":1,
    "Lucian":2,
    "Lulu":5,
    "Lux":3,
    "Malphite":1,
    "MasterYi":3,
    "MissFortune":5,
    "Mordekaiser":2,
    "Neeko":3,
    "Poppy":1,
    "Rakan":2,
    "Rumble":3,
    "Shaco":3,
    "Shen":2,
    "Sona":2,
    "Soraka":4,
    "Syndra":3,
    "Thresh":5,
    "TwistedFate":1,
    "VelKoz":4,
    "Vi":3,
    "WuKong":4,
    "Xayah":1,
    "Xerath":5,
    "XinZhao":2,
    "Yasuo":2,
    "Ziggs":1,
    "Zoe":1 
}

sample1_df["unit_cost"] = sample1_df["champions"].map(cost_dict)
sample1_df["comb_cost"] = 3 ** (sample1_df["star"]-1) * sample1_df["unit_cost"]        
            """)
    
    st.title("Evaluating Win/Loss")
    st.code("""
sample1_df["win"] = np.where(sample1_df["Ranked"] <= 4, "win", "lose")            
            """)
            
    st.title("Making the Simplified Dataframe")
    st.subheader("After grouping, the dataframe reached 44M+ rows. This contained many repetitions and made calculations incredibly slow. THerefore, we need to slim down the dataset, and remove the repeating rows.")
    st.code("""
    traits_dict = {
    "Ahri":["StarGuardian", "Set3_Sorcerer"],
    "Annie":["MechPilot","Set3_Sorcerer"],
    "Ashe":["Set3_Celestial","Sniper"],
    "AurelionSol":["Rebel","Starship"],
    "Blitzcrank":["Chrono","Set3_Brawler"],
    "Caitlyn":["Chrono","Sniper"],
    "ChoGath":["Set3_Void","Set3_Brawler"],
    "Darius":["SpacePirate","ManaReaver"],
    "Ekko":["Infiltrator","Cybernetic"],
    "Ezreal":["Chrono","Blaster"],
    "Fiora":["Cybernetic","Set3_Blademaster"],
    "Fizz":["MechPilot","Infiltrator"],
    "Gangplank":["SpacePirate","Demolitionist","Mercenary"],
    "Graves":["SpcaePirate","Blaster"],
    "Irelia":["Cybernetic","Set3_Blademaster","ManaReaver"],
    "JarvanIV":["DarkStar","Protector"],
    "Jayce":["SpcaePirate","Vanguard"],
    "Jhin":["DarkStar","Sniper"],
    "Jinx":["Rebel","Blaster"],
    "KaiSa":["Valkyrie","Infiltrator"],
    "Karma":["DarkStar","Set3_Mystic"],
    "Kassadin":["Set3_Celestial","ManaReaver"],
    "Kayle":["Valkyrie","Set3_Blademaster"],
    "KhaZix":["Set3_Void","Infiltrator"],
    "Leona":["Cybernetic","Vanguard"],
    "Lucian":["Cybernetic","Blaster"],
    "Lulu":["Set3_Celestial","Set3_Mystic"],
    "Lux":["DarkStar","Set3_Sorcerer"],
    "Malphite":["Rebel","Set3_Brawler"],
    "MasterYi":["Rebel","Set3_Blademaster"],
    "MissFortune":["Valkyrie","Mercenary","Blaster"],
    "Mordekaiser":["DarkStar","Vanguard"],
    "Neeko":["StarGuardian","Protector"],
    "Poppy":["StarGuardian","Vanguard"],
    "Rakan":["Set3_Celestial","Protector"],
    "Rumble":["MechPilot","Demolitionist"],
    "Shaco":["DarkStar","Infiltrator"],
    "Shen":["Chrono","Set3_Blademaster"],
    "Sona":["Rebel","Set3_Mystic"],
    "Soraka":["StarGuardian","Set3_Mystic"],
    "Syndra":["StarGuardian","Set3_Sorcerer"],
    "Thresh":["Chrono","ManaReaver"],
    "TwistedFate":["Chrono","Set3_Sorcerer"],
    "VelKoz":["Set3_Void","Sorcerer"],
    "Vi":["Cybernetic","Set3_Brawler"],
    "WuKong":["Chrono","Vanguard"],
    "Xayah":["Set3_Celestial","Set3_Blademaster"],
    "Xerath":["DarkStar","Set3_Sorcerer"],
    "XinZhao":["Set3_Celestial","Protector"],
    "Yasuo":["Rebel","Set3_Blademaster"],
    "Ziggs":["Rebel","Demolitionist"],
    "Zoe":["StarGuardian","Set3_Sorcerer"] 
}

champion_traits_list = [(champion, trait) for champion, traits in traits_dict.items() for trait in traits]

champion_traits_df = pd.DataFrame(champion_traits_list, columns=["champions", "traits"])

# Merge the two DataFrames based on the "champion" column
filtered_df = test.merge(champion_traits_df, on=["champions", "traits"], how="inner")""")
    
    st.subheader("Using df.merge on a dataframe of know champion-trait combinations, we can find the rows that were correct. At this point, the dataframe is at 10M+ rows. There are still multiple repetitions in the dataframe, so we need to clean it one more time to reach the desired result.")
    st.code("""
    filtered_df["win"] = np.where(filtered_df["Ranked"] <= 4, "win", "lose")
final_final_df = filtered_df[filtered_df.duplicated() == False]
final_final_df = final_final_df.sort_values(["gameId", "playerID"])     
            """)
    st.subheader("The dataframe now only has 1M+ rows, and is suitable for use.")
    st.title("The Final Dataframe")
    st.dataframe(filtered.head(20))

if selected == "Data Analysis":
    st.title("Data Analysis")
    st.markdown("The aim of the study is to find the best model to predict the 'win' variable, which indicates whether or not a player will rank in the top 4 of a single game. The goal is to create a Naive Bayes model and to identify useful predictors for the model to generate the highest accuracy when predicting win / loss.")
    st.header("Baseline Accuracy")
    st.markdown("The baseline model is a model used to act as a reference to the main model. It is the result expected when the minimum amount of calculation is used. The baseline accuracy can be found using pd.value_counts(normalize = True): ")
    st.dataframe(filtered.win.value_counts(normalize = True))
    st.markdown("In this case, the baseline accuracy is 52.19%. Therefore, we have a 52.19% chance of being accurate when we blindly predict a win. We would want the model to be more accurate than that.")
    st.header("The Naive Bayes Model")
    st.markdown("The Naive Bayes model is a simple probability classifier. By finding the posterior probability of each class, we can find the general probability and the prediction is assigned to the class with the highest probability. This model is mainly used in real-life classification, such as spam protection and sentiment analysis. A major limitation of this model is that we assume the probability every variable to be independent, which is unrealistic as many variables in a dataset are often dependent on one another.")
    st.header("Chi-Square Value")
    def chi2(item):
        chi2 = []
        p_val = []
        for x in item:
            chi2.append(stats.chi2_contingency(pd.crosstab(filtered["win"], filtered[x]))[0])
            p_val.append(stats.chi2_contingency(pd.crosstab(filtered["win"], filtered[x]))[1])
        return chi2, p_val  
                                      
    chi_square = pd.DataFrame({"name":['level', "traits", "num_traits", "champions", 'star', "unit_cost"]})
    chi_square["chi_square"], chi_square["Pvalue"] = chi2(['level', "traits", "num_traits", "champions", 'star', "unit_cost"])
    st.dataframe(chi_square.sort_values("chi_square", ascending = False))
    st.markdown("Above is every categorical variable, and their respective chi square value and P-value when compared to 'win'. The chi-square indicator shows a variable's relevance to the variable we want to analyse. Therefore, this dataframe will help determine which variables are the best to choose as predictors to improve the model's accuracy.")
    st.markdown("A large chi-square value means that there is a stronger association between variables. In this case, 'level' has the highest chi-square value, meaning that it is closely associated with 'win'. Therefore, 'level' is a good predictor to use for the model. The P-value can also be used to show relevance, as smaller values signify a more relevant variable. However, as all of the P-values listed above are 0.00, it is difficult to use the P-values to select predictors.")
    st.header("Function Testing")
    st.markdown("We can first try to put 'level' through the function as the only predictor:")
    st.code("classifier(filtered, ['level'], 'win')")
    r, a = classifier(filtered, ['level'], 'win')
    st.dataframe(r.head(20))
    st.markdown(f"The accuracy of using 'level' as the sole predictor is {np.round(a*100, 2)}%. However, this is quite close to the baseline accuracy - meaning that 'level' cannot be the only predictor used in the model.")
    fig_igd = px.box(filtered, y = "ingameDuration", color = "win", color_discrete_sequence=px.colors.qualitative.Dark2,labels = names)
    fig_igd.update_layout(legend_title_text="")
    fig_cc = px.box(filtered, y = "comb_cost", color = "win", color_discrete_sequence=px.colors.qualitative.Dark2,labels = names)
    fig_cc.update_layout(legend_title_text="")
    fig_igd.update_layout(title="In-game Duration Against Win / Loss")
    fig_cc.update_layout(title="Total Cost Against Win / Loss")
    coli, colc = st.columns([5,5])
    coli.plotly_chart(fig_igd)
    colc.plotly_chart(fig_cc)
    st.markdown("There are only two numeric variables - 'ingameDuration' and 'comb_cost'. By plotting box charts, we can look at the mean difference of these variables between win and loss. It seems that ingameDuration has a higher mean difference, meaning that it is a better predictor as it is more relevant to 'win'.")
    st.markdown("The highest accuracy can be found through:")
    st.code("classifier(filtered, ['ingameDuration','level','traits','unit_cost','star'], 'win')")
    st.markdown("at 85.7226%. This is a great improvement over the baseline accuracy of 52.19% and shows that this is a successful model. These columns are mainly the ones with the highest chi-square scores, although 'champions' is not included as it is too complex and can lower the accuracy. However, it is still represented in the model through more general categorisation like 'unit_cost'. ")
    st.markdown("Below is a scatterplot of every player, their average cost per piece, their list of champions and their game time. Alongside that is every champion & star to have appeared in a winning combination.")
    c_box, t_box = st.columns([5,5])
    fig_cmb = px.scatter(players_champs, x = "comb_cost", y = "ingameDuration", color = "win", color_discrete_sequence=px.colors.qualitative.Dark2, hover_data = ["champions"],labels = names)
    fig_cmb.update_layout(title="Scatterplot of Individual Players")
    c_box.plotly_chart(fig_cmb)
    fig_cmx = px.scatter(players_traits, x = "ingameDuration", y = "win", color = "star", color_discrete_sequence=px.colors.qualitative.Dark2, hover_data = ["champions", "comb_cost"],labels = names)
    fig_cmx.update_layout(title="Scatterplot of Combinations")
    fig_cmx.update_xaxes(title_text="Average of In-game Duration")
    fig_cmx.update_yaxes(title_text="Win Count")
    t_box.plotly_chart(fig_cmx)
    
    st.markdown("Champion Histogram")
    fig_ch = px.histogram(filtered, x = "champions", facet_col = "win", facet_row_spacing = 0.2, facet_col_wrap = 1, histnorm = "percent", height = 800, width = 1000,labels = names).update_xaxes(categoryorder='total descending', showticklabels = True)
    fig_ch.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].title()))
    st.plotly_chart(fig_ch)
    fig_sb_win = px.sunburst(filtered[filtered["win"] == "Win"], path = ["champions", "traits"], width = 800, height = 1000,labels = names)
    st.plotly_chart(fig_sb_win)
    fig_sb_lose = px.sunburst(filtered[filtered["win"] == "Lose"], path = ["champions", "traits"], width = 800, height = 1000,labels = names)
    st.plotly_chart(fig_sb_lose)
    st.markdown("Looking at the two sunburst plots for win and loss, similar champions are both used by the majority of players, aside of win and loss. Therefore, it is possible to conclude that traits such as 'chrono' and 'brawler' are viewed as beneficial and attractive amongst players. This also shows that most elimination occurs in the mid-game, as many are able to access high cost pieces such as Miss Fortune but cannot successfully develop the same trait combinations against competitors.")
    
if selected=='Exploratory Analysis':
    st.title("Exploratory Analysis")
    st.header("Interactive Histogram")
    
    col1, col2 = st.columns([2,8])
    with st.form("Interactive Histogram"):
        x_option=col1.selectbox('Select a column for x:',filtered.columns.sort_values(), key = 2)
        y_option=col1.selectbox('Select a column for y:',np.setdiff1d(filtered.columns.sort_values(), x_option), key = 4)
        colors = filtered.columns.sort_values().insert(0, None)
        color_option=col1.selectbox('Select a column for the color:',colors, key = 9)
        barmode_option = col1.selectbox('Select a option for the barmode:',[None, 'group', 'overlay','relative'], key = 5)
        barnorm_option = col1.selectbox('Select a option for the barnorm:',[None, 'fraction', 'percent'], key = 6)
        histnorm_option = col1.selectbox('Select a option for the histnorm:',[None, 'percent', 'probability', 'density', 'probability density'], key = 7)
        histfunc_option = col1.selectbox('Select a option for the histfunc:',[None, 'count', 'sum', 'avg', 'min', 'max'], key = 8)
        submitted=st.form_submit_button("Submit to generate a histogram: ")
        
        if submitted:
            fig = px.histogram(filtered, x = x_option, y = y_option, color = color_option, barmode = barmode_option, barnorm = barnorm_option, histnorm = histnorm_option, histfunc = histfunc_option,labels = names)
            col2.plotly_chart(fig)
            
    st.header("Interactive Histogram (Facet)")
    
    col3, col4 = st.columns([2,8])
    with st.form("Interactive Histogram (Facet)"):
        x_option2=col3.selectbox('Select a column for x:',filtered.columns.sort_values(), key = 17)
        y_option2=col3.selectbox('Select a column for y:',filtered.columns.sort_values(), key = 32)
        colors2 = filtered.columns.sort_values().insert(0, None)
        color_option2=col3.selectbox('Select a column for the color:',colors, key = 19)
        barmode_option2 = col3.selectbox('Select a option for the barmode:',[None, 'group', 'overlay','relative'], key = 15)
        barnorm_option2 = col3.selectbox('Select a option for the barnorm:',[None, 'fraction', 'percent'], key = 12)
        histnorm_option2 = col3.selectbox('Select a option for the histnorm:',[None, 'percent', 'probability', 'density', 'probability density'], key = 11)
        histfunc_option2 = col3.selectbox('Select a option for the histfunc:',[None, 'count', 'sum', 'avg', 'min', 'max'], key = 1984)
        facet = filtered.columns.sort_values().insert(0, None)
        facet_option = col3.selectbox('Select a option for the facet column:',filtered.columns.sort_values(), key = 10)
        facet_col_option = st.slider(label = 'facet_col_wrap', min_value = 1, max_value = 5)
        submitted2=st.form_submit_button("Submit to generate a histogram: ")
        
        if submitted2:
            fig2 = px.histogram(filtered, x = x_option2, y = y_option2, color = color_option2, barmode = barmode_option2, barnorm = barnorm_option2, histnorm = histnorm_option2, histfunc = histfunc_option2, facet_col = facet_option, facet_col_wrap = facet_col_option, width = 1000,labels = names)
            fig2.for_each_yaxis(lambda y: y.update(title = ''))
            fig2.add_annotation(x=-0.06,y=0.5,
                   text=y_option, textangle=-90,
                    xref="paper", yref="paper", showarrow=False)
            col4.plotly_chart(fig2)
            
    st.header('Interactive Box Plot -- Comparing Variables to "win"')
    col5, col6 = st.columns([3,5])
    with st.form('Interactive Box Plot -- Comparing Variables to "win"'):
        var_option=col5.selectbox('Select a variable:',filtered.columns.sort_values(), key = 123)
        submitted3=st.form_submit_button("Submit to generate a box plot: ")
        
        if submitted3:
            fig3 = px.box(filtered, y = var_option, color = "win",labels = names)
            col6.plotly_chart(fig3)
            
if selected=='Interactive Naive Bayes':
    st.title("Interactive Naive Bayes Predictions")
    st.subheader("Select the parameters for the function here:")
    with st.form("test"):
        coln, colm = st.columns([3, 7])
        predictors=coln.multiselect("Select the parameters you wish to use",np.setdiff1d(list(filtered.columns.sort_values()), ["win", "Ranked", "items","playerID", "gameId" ]),key=20,default=None)
        submitted4=st.form_submit_button("Submit to view accuracy")
        if submitted4:
            result, accuracy = classifier(filtered, predictors, "win")
            st.subheader(f"The accuracy is {np.round(accuracy*100,4)}%")
            st.dataframe(result.head(20))
            cm = confusion_matrix(result["Given"], result["Pred"])
            fig4 = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), text_auto = True)
            fig4.update_layout(title = "Confusion Matrix: Given Against Predicted")
            st.plotly_chart(fig4, use_container_width=True)
            st.balloons()
            
if selected =="Conclusion":
    st.title("Conclusion")
    st.markdown("In conclusion, the factors that are the best in predicting winning results are 'ingameDuration','level','traits','unit_cost' and 'star'. Variables such as ingameDuration and level are incredibly useful in predicting winning results as higher ranked players tend to have a higher game duration and higher level. Similar logics apply to unit_cost and star as if winning players have a higher level, they gain more access to higher-cost pieces and they can make higher stars with them.")
    st.markdown("In selecting a team that is likely to win, a player should pick a trait that has a high win rate and try to build it from the start of the game. Try to upgrade core pieces and supply useful items. With this strategy one is highly likely to win if performed properly. Other tactics are often avaliable (e.g. collection of first-level traits) but they carry more risk. There is no definitive combination to winnign the game, but the player can always choose from their personal preference which tactic is the most fun to play with.")