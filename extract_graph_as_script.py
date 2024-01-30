# %% [markdown]
# ## Setup

# %% [markdown]
# Run in terminal:
# pip install typing-extensions<4.6.0
# pip install pillow<10.1.0,>=8.3.2
# pip install fastapi kaleido uvicorn
# pip install langchain
# pip install pypdf
# pip install unstructured
# pip install yachalk
# pip install "unstructured[pdf]"
# pip install openai
# sudo apt update
# sudo apt-get install libgl1-mesa-glx
# 
# pip install --upgrade jupyter ipywidgets
# 
# pip install --upgrade opencv-python-headless
# 
# export OPENAI_API_KEY="..."
# 

# %%
import pandas as pd
import numpy as np
import os
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random

## Input data directory
data_dir = "coin"
inputdirectory = Path(f"./data_input/{data_dir}")
## This is where the output csv files will be written
out_dir = data_dir
outputdirectory = Path(f"./data_output/{out_dir}")


# %%

## Dir PDF Loader
loader = PyPDFDirectoryLoader(inputdirectory)
## File Loader
#loader = PyPDFLoader("./data_input/coin/coinbase.pdf")
#loader = DirectoryLoader(inputdirectory, show_progress=True)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

pages = splitter.split_documents(documents)
print("Number of chunks = ", len(pages))
print(pages[3].page_content)

# %% [markdown]
# ## Create a dataframe of all the chunks

# %%
from helpers.df_helpers import documents2Dataframe
df = documents2Dataframe(pages)
df.to_csv("df.csv", sep="|", index=False)
print(df.shape)
df.head()

# %% [markdown]
# ## Extract Concepts

# %%

from openai import OpenAI
import pandas as pd
import numpy as np
import openai
import json
import os

client = OpenAI(organization='org-A6mbTbr0FP5rFIEvwzMViNHR')
selected_model = "gpt-4-0125-preview"

def graphPrompt(input: str, metadata={}, model=selected_model):
    openai.api_key = os.getenv("OPENAI_API_KEY") 


    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    )

    USER_PROMPT = f"context: ```{input}``` \n\n output: "
   # response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
            
    response = client.chat.completions.create(
    model=model,
    response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
    )
    try:

        content = response.choices[0].message.content
        #print('content', content)
        
        # Parse the JSON string
        parsed_content = json.loads(content)

        # Handle single object or list
        if isinstance(parsed_content, dict):
            result = [dict(parsed_content, **metadata)]
        elif isinstance(parsed_content, list):
            result = [dict(item, **metadata) for item in parsed_content]
        else:
            raise ValueError("Unexpected data format in response")

        #print('result', result)
    except Exception as e:
        print("\n\nERROR ### Here is the errored response: ", response, "\n\nException: ", e)
        result = None
    print('graph prompt function success')
    return result

def process_chunk(chunk, model):
    chunk.reset_index(inplace=True)
    results = chunk.apply(lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}, model), axis=1)
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)
    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    print('chunk result generated')
    return concept_list


def df2Graph(dataframe: pd.DataFrame, model=None) -> list:
    chunk_size = 5  # Processing in chunks of 20 rows
    chunks = [dataframe[i:i + chunk_size] for i in range(0, dataframe.shape[0], chunk_size)]
    chunkstotal = len(chunks)
    print('chunks total of ', chunkstotal)
    all_concepts = []
    counter=0
    for chunk in chunks:
        counter=counter+1
        print(f'df2Graph started chunk #{counter}/{chunkstotal}')
        chunk_concepts = process_chunk(chunk, model)
        print(f'df2Graph finished chunk #{counter}/{chunkstotal}')
        if chunk_concepts:  # Check if the chunk_concepts is not None
            all_concepts.extend(chunk_concepts)
        else:
            continue
    print(f'df2Graph finished all chunks')


    return all_concepts


def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())

    return graph_dataframe

# %%

def dfgcsv(dataframe: pd.DataFrame, model=None) -> pd.DataFrame:
    chunk_size = 5  # Processing in chunks of 20 rows
    chunks = [dataframe[i:i + chunk_size] for i in range(0, dataframe.shape[0], chunk_size)]
    chunkstotal = len(chunks)
    print('chunks total of ', chunkstotal)
    all_concepts = []
    counter=0
    for chunk in chunks:
        counter=counter+1
        print(f'df2Graph started chunk #{counter}/{chunkstotal}')
        chunk_concepts = process_chunk(chunk, model)
        print(f'df2Graph finished chunk #{counter}/{chunkstotal}')
        graph_dataframe = pd.DataFrame(chunk_concepts).replace(" ", np.nan)
        graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
        graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
        graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())
        graph_dataframe.to_csv(os.path.join(outputdirectory, "graph.csv"), sep="|", index=False, mode='a')
        chunk.to_csv(os.path.join(outputdirectory, "chunks.csv"), sep="|", index=False, mode='a')
    return

dfgcsv

# %%
df = pd.read_csv('df.csv', sep="|") 

# %% [markdown]
# If regenerate is set to True then the dataframes are regenerated and Both the dataframes are written in the csv format so we dont have to calculate them again. 
# 
#         dfne = dataframe of edges
# 
#         df = dataframe of chunks
# 
# 
# Else the dataframes are read from the output directory

# %%
regenerate = True
if regenerate:
    concepts_list = df2Graph(df, model="gpt-4-0125-preview")
    print('concept_list created')
    dfg1 = graph2Df(concepts_list)
    print('concept_list processed')
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)
    
    dfg1.to_csv(os.path.join(outputdirectory, "graph.csv"), sep="|", index=False)
    df.to_csv(os.path.join(outputdirectory, "chunks.csv"), sep="|", index=False)
    print('concept saved')
else:
    dfg1 = pd.read_csv(os.path.join(outputdirectory, "graph.csv"), sep="|")

dfg1.replace("", np.nan, inplace=True)
dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
dfg1['count'] = 4 
## Increasing the weight of the relation to 4. 
## We will assign the weight of 1 when later the contextual proximity will be calculated.  
print(dfg1.shape)
dfg1.head()

# %% [markdown]
# ## Calculating contextual proximity

# %%
def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    ## Melt the dataframe into a list of nodes
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2


dfg2 = contextual_proximity(dfg1)
dfg2.tail()

# %% [markdown]
# ### Merge both the dataframes

# %%
dfg = pd.concat([dfg1, dfg2], axis=0)
dfg = (
    dfg.groupby(["node_1", "node_2"])
    .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
    .reset_index()
)
dfg

# %% [markdown]
# ## Calculate the NetworkX Graph

# %%
nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
nodes.shape

# %%
import networkx as nx
G = nx.Graph()

## Add nodes to the graph
for node in nodes:
    G.add_node(
        str(node)
    )

## Add edges to the graph
for index, row in dfg.iterrows():
    G.add_edge(
        str(row["node_1"]),
        str(row["node_2"]),
        title=row["edge"],
        weight=row['count']/4
    )

# %% [markdown]
# ### Calculate communities for coloring the nodes

# %%
communities_generator = nx.community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
communities = sorted(map(sorted, next_level_communities))
print("Number of Communities = ", len(communities))
print(communities)

# %% [markdown]
# ### Create a dataframe for community colors

# %%
import seaborn as sns
palette = "hls"

## Now add these colors to communities and make another dataframe
def colors2Community(communities) -> pd.DataFrame:
    ## Define a color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors


colors = colors2Community(communities)
colors

# %% [markdown]
# ### Add colors to the graph

# %%
for index, row in colors.iterrows():
    G.nodes[row['node']]['group'] = row['group']
    G.nodes[row['node']]['color'] = row['color']
    G.nodes[row['node']]['size'] = G.degree[row['node']]

# %%
from pyvis.network import Network

graph_output_directory = "./docs/index.html"

net = Network(
    notebook=False,
    # bgcolor="#1a1a1a",
    cdn_resources="remote",
    height="900px",
    width="100%",
    select_menu=True,
    # font_color="#cccccc",
    filter_menu=False,
)

net.from_nx(G)
# net.repulsion(node_distance=150, spring_length=400)
net.force_atlas_2based(central_gravity=0.015, gravity=-31)
# net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380)
net.show_buttons(filter_=["physics"])

net.show(graph_output_directory, notebook=False)

# %%



