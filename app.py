from tabnanny import check
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input,Output,State
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
from urllib.request import urlopen
import dash_table_experiments as dt
import io
import json
import numpy as np
from collections import defaultdict,Counter
from nltk.corpus import stopwords
import dash.dependencies as dd
from joblib import load
from io import BytesIO
import tensorflow as tf
import pandas as pd
from wordcloud import WordCloud
import base64
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification


filename = "./model"

loaded_model = TFDistilBertForSequenceClassification.from_pretrained(filename)
tokenizer = DistilBertTokenizerFast.from_pretrained(filename)
# prediction function and load model
def ValuePredictor(to_predict):
    predictions = []

    for x in to_predict:
        mek = tokenizer.encode(x,truncation=True,padding=True,return_tensors="tf")
        j = loaded_model.predict(mek)[0]
        tf_prediction = tf.nn.softmax(j, axis=1)
        label = tf.argmax(tf_prediction, axis=1)
        label = label.numpy()[0]
        predictions.append(label)
        
    return predictions




#non stop word corpus
stop=set(stopwords.words('english'))
def create_corpus(target,data):
    corpus=[]
    
    for x in data[data.iloc[:, 1]==target ].iloc[:, 0].str.split():
        for i in x:
            if i not in stop:
                corpus.append(i)
    return corpus




app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server


# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "black",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


### SECTION TWO - LAYOUT ##


sidebar = html.Div(
    [
        #html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "OVERVIEW", className="lead",style={'color':'white'}
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Dashboard", href="/page-1", active="exact"),

            ],
            vertical=True,
            pills=True,
        ), 
        
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])


### SECTION 3 - Callbacks ###

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")])

def render_page_content(pathname):

    if pathname == "/":
        return [
               
    html.Div([
            html.Img(src=app.get_asset_url('emotions.png'),
                     id = 'jalurgemilang-image',
                     style={'height': '60px',
                            'width': 'auto',
                            'margin-bottom': '25px',
                            'display' : 'inline'
}),
    

html.H1('STUDENT FEEDBACK SENTIMENT ANALYSIS',
style = {
    'display' : 'inline',
    'textAlign':'center',
    'color' : 'white',
    'margin-left': '25px',
    'fontFamily': 'Roboto Mono',
    'fontSize': 30

}),


#area to enter text
 dbc.Textarea(id='textarea', className="mb-3", placeholder="Enter a review and find out it's sentiment!",
                                 value='', style={'resize': 'none'}),

#result of sentiment
html.Div(id='result'),




],style={'textAlign':'center'}),



html.Div(dcc.Markdown('''
            &nbsp;  
            &nbsp;  
            Built by [Aisyah Razak (aisyahrazak171@gmail.com)](https://www.linkedin.com/in/aisyahh-razak/)  
            View [App Source] (https://github.com/aisyahrzk/covid-19-dashboard)
            '''),
            style={
                'textAlign': 'center',
                'color': 'white',
                'width': '100%',
                'float': 'center',
                'display': 'inline-block'}
            )

                
            ]


    elif pathname == "/page-1":
        return [

            html.H1('Generate your own dashboard!',
style = {
    'display' : 'inline',
    'textAlign':'center',
    'color' : 'white',
    'margin-left': '25px',
    'fontFamily': 'Roboto Mono',
    'fontSize': 30

}),
   
html.Div([
    dcc.Upload(
        id='upload-data', 
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'color':'white',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),


    dcc.Store(id = 'uploaded-data',storage_type='session' ),
    
    html.Div(id = 'big-value',className='row flex-display'),

   html.Div([

dcc.Graph(id ='pie-custom',config={'displayModeBar': 'hover'}

)], style={'width': '100%'}, className='create_container four columns'),

 html.Div(dcc.RadioItems(id='dropdown_Sentiment',
            options=[{'label': i, 'value': i} for i in ['Positive', 'Negative','Neutral']],
            value='Positive',labelStyle={'float': 'center', 'display': 'inline-block'},
            inputStyle={"margin-right": "20px"}
            ), style={'textAlign': 'center',
                'color': 'white',
                'width': '100%',
                'float': 'center',
                'display': 'inline-block',
                'margin-right':2
            }
           
        ),

        html.Div([

dcc.Graph(id ='output-wordfreq',config={'displayModeBar': 'hover'}

)], style={'width': '1500'}, className='create_container five columns'),
 
  
 
]),

html.Div([

    html.Button("Download CSV File of labelled reviews!", id="btn_csv"),
        dcc.Download(id="download-dataframe-csv"),

])



    ]

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("btn_csv", "n_clicks"),
    Input("uploaded-data","data")],
    prevent_initial_call=True,
)
def func(n_clicks,data):

    if n_clicks:
        df = json.loads(data)
        df = pd.DataFrame.from_dict(df, orient="columns")
        return dcc.send_data_frame(df.to_csv, "labeled.csv")






@app.callback(
    Output('result', 'children'),
    [
        Input('textarea', 'value')
    ],
)

def result_predict(value):

    prediction = 4

    if value!='':
        prediction = ValuePredictor([value])[0]

    if (prediction == 1):
        return dbc.Alert("Ahh too much Negativity!", color="danger")
    elif (prediction == 0):
        return dbc.Alert("Positive!", color="success")
    elif (prediction == 2):
        return dbc.Alert("Neutral", color="secondary")
    else:
        return dbc.Alert("Write something :)", color="info")
 
        

        

def parse_contents(contents, filename):
    
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:

        return html.Div([
            'There was an error processing this file.'
        ])

    return df



@app.callback(Output('output-wordfreq', 'figure'),
              [Input('uploaded-data', 'data'),
              Input('dropdown_Sentiment', 'value')])
def update_output(data,sentiment):

    

    if data:
        
        df_j = json.loads(data)
        
        df = pd.DataFrame.from_dict(df_j, orient="columns")
    


    if sentiment == 'Positive':
        dic=defaultdict(int)
        corpus = create_corpus(0,df)
        
    
        for word in corpus:
            dic[word]+=1
        top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
            
        x,y=zip(*top)
            
        color = '#17C37B'
        name_title = 'Word Frequency in Positive Reviews'
        
        
    elif sentiment == 'Negative':
        dic=defaultdict(int)
        corpus = create_corpus(1,df)
    
        for word in corpus:
            dic[word]+=1
        top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
        x,y=zip(*top)

        color = '#FF0000'
        name_title = 'Word Frequency in Negative Reviews'

    elif sentiment == 'Neutral':
        dic=defaultdict(int)
        corpus = create_corpus(2,df)
    
        for word in corpus:
            dic[word]+=1
        top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
        x,y=zip(*top)

        color = '#808080'
        name_title = 'Word Frequency in Neutral Reviews'

    
    return{
            
        'data': [go.Bar(
            x=x,
            y=y,
            name=name_title,
            marker=dict(color=color)),
        ],


            'layout': go.Layout(
            title={'text': name_title,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            titlefont={'color': 'white',
                       'size': 20},
            font=dict(family='Roboto Mono',
                      color='white',
                      size=12),
            hovermode='closest',
            paper_bgcolor='#1f2c56',
            plot_bgcolor='#1f2c56',
            legend={'orientation': 'h',
                    'bgcolor': '#1f2c56',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.7},
            margin=dict(r=0),
            xaxis=dict(title='<b>Word</b>',
                       color = 'white',
                       showline=True,
                       showgrid=True,
                       showticklabels=True,
                       linecolor='white',
                       linewidth=1,
                       ticks='outside',
                       tickfont=dict(
                           family='Roboto Mono',
                           color='white',
                           size=12
                       )),
            yaxis=dict(title='<b>Word Frequency</b>',
                       color='white',
                       showline=True,
                       showgrid=True,
                       showticklabels=True,
                       linecolor='white',
                       linewidth=1,
                       ticks='outside',
                       tickfont=dict(
                           family='Roboto Mono',
                           color='white',
                           size=12
                       ))
                       )
    }



@app.callback(Output('uploaded-data', 'data'), [Input('upload-data', 'contents'),
              Input('upload-data', 'filename')])

def predict_data(list_content,filename):
     
     if list_content:
        
        contents = list_content[0]
        filename = filename[0]

        df = parse_contents(contents, filename)

        prediction = ValuePredictor(df.iloc[:,0])




        df['label'] = prediction


     return df.to_json(date_format='iso', orient='columns')



@app.callback(Output('pie-custom', 'figure'),
              Input('uploaded-data', 'data'))
def update_pie(data):
        
    if data:
        df_j = json.loads(data)
        df = pd.DataFrame.from_dict(df_j, orient="columns")


    positive3 = len(df[df.iloc[:, 1]==0])
    negative3 = len(df[df.iloc[:, 1]==1])
    neutral3 = len(df[df.iloc[:, 1]==2])


    return  {'data': [go.Pie(
            labels=['Positive', 'Negative', 'Neutral'],
            values=[positive3,negative3,neutral3],
            marker=dict(colors= ['chartreuse', 'red', 'blue']),
            hoverinfo='value',
            textinfo='label+percent',
            texttemplate = "%{label}:%{percent}",
            hole=.7,
            rotation=45,

        )],

        'layout': go.Layout(
            title={'text': 'Reviews: ' + 'Classification',
                   'y': 0.93,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            titlefont={'color': 'white',
                       'size': 20},
            font=dict(family='Roboto Mono',
                      color='white',
                      size=12),
            hovermode='closest',
            paper_bgcolor='#1f2c56',
            legend={'orientation': 'h',
                    'bgcolor': '#1f2c56',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.7})
                    
                }



@app.callback(Output('big-value', 'children'),
              Input('uploaded-data', 'data'))
def update_marker(data):

    if data:

        df_j = json.loads(data)
        
        df = pd.DataFrame.from_dict(df_j, orient="columns")


    positive3 = len(df[df.iloc[:, 1]==0])
    negative3 = len(df[df.iloc[:, 1]==1])
    neutral3 = len(df[df.iloc[:, 1]==2])

    return [

html.Div([
        html.Div([
    
   html.H6(children='TOTAL REVIEWS',
                    style={'textAlign': 'center',
                           'color': 'white'}),
            html.P(f"{len(df.index):,.0f}",
                    style={'textAlign': 'center',
                           'color': 'orange',
                           'fontSize': 40})
       
        ], style={'width': '21%'},className='card_container three columns'),

html.Div([
    
    html.H6(children='POSITIVE REVIEWS',
                    style={'textAlign': 'center',
                           'color': 'white'}),
            html.P(f"{positive3:,.0f}",
                    style={'textAlign': 'center',
                           'color': 'green',
                           'fontSize': 40})
       
        ], style={'width': '21%'},className='card_container three columns'),

html.Div([
#no data yet
    html.H6(children='NEGATIVE REVIEWS',
                    style={'textAlign': 'center',
                           'color': 'white'}),
            html.P(f"{negative3:,.0f}",
                    style={'textAlign': 'center',
                           'color': 'red',
                           'fontSize': 40})
       
        ], style={'width': '21%'},className='card_container three columns'),

            
html.Div([
 html.H6(children='NEUTRAL REVIEWS',
                    style={'textAlign': 'center',
                           'color': 'white'}),
            html.P(f"{neutral3:,.0f}",
                    style={'textAlign': 'center',
                           'color': 'gray',
                           'fontSize': 40})
       
        ], style={'width': '21%'},className='card_container three columns'),
        ],style = {'margin':'auto'},className='row flex-display')
    ]



  

        
# automatically update HTML display if a change is made to code
if __name__ == '__main__':
    
    app.server.run(debug=True)