import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Carregar o modelo e os objetos de pré-processamento
opt_gb = joblib.load('opt_gb_model.pkl')
encoder = joblib.load('onehot_encoder.pkl')
scaler = joblib.load('minmax_scaler.pkl')

# Funções de pré-processamento
def apply_encoder(dataframe, encoder):
    categoric_columns = ['Gender', 'Platform']
    categoric_data = dataframe[categoric_columns]
    data_encoded = pd.DataFrame(encoder.transform(categoric_data), columns=encoder.get_feature_names_out(categoric_columns))
    final_data = pd.concat([dataframe.drop(categoric_columns, axis=1), data_encoded], axis=1)
    return final_data

def normalize(dataframe, scaler):
    columns_to_normalize = [
        'Age',
        'Daily_Usage_Time (minutes)',
        'Posts_Per_Day',
        'Likes_Received_Per_Day',
        'Comments_Received_Per_Day',
        'Messages_Sent_Per_Day'
    ]
    normalized_data = dataframe.copy()
    normalized_data[columns_to_normalize] = scaler.transform(normalized_data[columns_to_normalize])
    return normalized_data

def preprocess(dataframe):
    # Remover a coluna 'User_ID' se existir
    if 'User_ID' in dataframe.columns:
        dataframe = dataframe.drop(columns=['User_ID'])
    
    encoded_data = apply_encoder(dataframe, encoder)
    normalized_data = normalize(encoded_data, scaler)
    return normalized_data

# Interface do Streamlit
st.set_page_config(page_title='Análise de Emoções nas Redes Sociais', layout='wide')
st.title('Previsão de Emoções Dominantes nas Redes Sociais')

# Carregar o arquivo automaticamente
data = pd.read_csv('dataset.csv')
st.subheader("Dados Carregados")
st.write(data.head())

# Remover a coluna 'User_ID' se existir
if 'User_ID' in data.columns:
    data = data.drop(columns=['User_ID'])

# Resumo estatístico dos dados
st.subheader("Resumo Estatístico")
st.write(data.describe().drop('count'))

# Identificar colunas numéricas
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

# Pré-processar os dados
preprocessed_data = preprocess(data)
X = preprocessed_data.drop('Dominant_Emotion', axis=1, errors='ignore')

# Fazer previsões
predictions = opt_gb.predict(X)

# Adicionar previsões aos dados
data['Predicted_Emotion'] = predictions

# Mostrar os resultados
st.subheader("Dados com Previsões")
st.write(data)

# Selecionar uma amostra dos dados para visualização
sample_data = data.sample(n=100, random_state=1)

# Gráfico de barras comparando previsões com emoções reais
st.subheader("Comparação entre Emoções Reais e Previstas")
emotion_comparison = sample_data[['Dominant_Emotion', 'Predicted_Emotion']].reset_index().melt(id_vars=['index'], var_name='Type', value_name='Emotion')
fig_comparison = px.bar(emotion_comparison, x='index', y='Emotion', color='Type', barmode='group', title='Comparação entre Emoções Reais e Previstas')
st.plotly_chart(fig_comparison, use_container_width=True)

# Análise 1: Análise de Uso Diário por Plataforma
st.subheader("Análise de Uso Diário por Plataforma")
daily_usage_mean = data.groupby('Platform')['Daily_Usage_Time (minutes)'].mean().reset_index()
fig1 = px.bar(daily_usage_mean, x='Platform', y='Daily_Usage_Time (minutes)', title='Média de Tempo Diário por Plataforma', color='Platform', color_discrete_sequence=px.colors.qualitative.Pastel, height=600)
st.plotly_chart(fig1, use_container_width=True)

# Análise 2: Comparação de Emoções Dominantes por Gênero e Plataforma
st.subheader("Comparação de Emoções Dominantes por Gênero e Plataforma")
platforms = data['Platform'].unique()
for platform in platforms:
    platform_data = data[data['Platform'] == platform]
    emotion_gender_platform = platform_data.groupby(['Gender', 'Dominant_Emotion']).size().reset_index(name='Count')
    fig2 = px.bar(emotion_gender_platform, x='Gender', y='Count', color='Dominant_Emotion', barmode='group', title=f'Emoções Dominantes por Gênero na Plataforma {platform}', color_discrete_sequence=px.colors.qualitative.Pastel, height=600)
    st.plotly_chart(fig2, use_container_width=True)

# Análise 4: Análise de Emoções Dominantes por Plataforma
st.subheader("Análise de Emoções Dominantes por Plataforma")
emotion_platform = data.groupby('Platform')['Dominant_Emotion'].value_counts().unstack().fillna(0)
fig4 = px.bar(emotion_platform, barmode='stack', title='Distribuição das Emoções Dominantes por Plataforma', color_discrete_sequence=px.colors.qualitative.Pastel, height=600)
st.plotly_chart(fig4, use_container_width=True)

# Análise 5: Análise de Postagens Diárias e Engajamento
st.subheader("Análise de Postagens Diárias e Engajamento")
col1, col2 = st.columns(2)
with col1:
    fig5_1 = px.scatter(data, x='Posts_Per_Day', y='Likes_Received_Per_Day', color='Platform', title='Postagens Diárias vs Likes Recebidos', color_discrete_sequence=px.colors.qualitative.Pastel, height=600)
    st.plotly_chart(fig5_1, use_container_width=True)
with col2:
    fig5_2 = px.scatter(data, x='Posts_Per_Day', y='Comments_Received_Per_Day', color='Platform', title='Postagens Diárias vs Comentários Recebidos', color_discrete_sequence=px.colors.qualitative.Pastel, height=600)
    st.plotly_chart(fig5_2, use_container_width=True)

# Análise 6: Análise de Uso por Idade
st.subheader("Análise de Uso por Idade")
data['Age_Group'] = pd.cut(data['Age'], bins=[20, 25, 30, 35, 40], labels=['20-25', '26-30', '31-35', '36-40'])
age_usage = data.groupby('Age_Group')['Daily_Usage_Time (minutes)'].mean().reset_index()
fig6 = px.pie(age_usage, values='Daily_Usage_Time (minutes)', names='Age_Group', title='Média de Tempo Diário por Faixa Etária')
st.plotly_chart(fig6, use_container_width=True)

# Análise 7: Relação de Crescimento entre Tempo Diário e Postagens, Likes e Comentários
st.subheader("Relação de Crescimento entre Tempo Diário e Postagens, Likes e Comentários")

# Crescimento: Tempo Diário vs Postagens
fig7_1 = px.scatter(data, x='Daily_Usage_Time (minutes)', y='Posts_Per_Day', trendline='ols', title='Crescimento: Tempo Diário vs Postagens', color_discrete_sequence=['#1f77b4'], height=600)
# Crescimento: Likes vs Comentários
fig7_2 = px.scatter(data, x='Likes_Received_Per_Day', y='Comments_Received_Per_Day', trendline='ols', title='Crescimento: Likes vs Comentários', color_discrete_sequence=['#aec7e8'], height=600)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig7_1, use_container_width=True)
with col4:
    st.plotly_chart(fig7_2, use_container_width=True)

# Crescimento: Postagens vs Likes
fig7_3 = px.scatter(data, x='Posts_Per_Day', y='Likes_Received_Per_Day', trendline='ols', title='Crescimento: Postagens vs Likes', color_discrete_sequence=['#ff7f0e'], height=600)
# Crescimento: Postagens vs Comentários
fig7_4 = px.scatter(data, x='Posts_Per_Day', y='Comments_Received_Per_Day', trendline='ols', title='Crescimento: Postagens vs Comentários', color_discrete_sequence=['#2ca02c'], height=600)

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(fig7_3, use_container_width=True)
with col6:
    st.plotly_chart(fig7_4, use_container_width=True)

st.markdown("### Interpretação das Relações de Crescimento")
st.markdown("""
- **Crescimento entre Tempo Diário e Postagens**: Uma linha de tendência foi adicionada para mostrar como o tempo diário de uso influencia a quantidade de postagens diárias.
- **Crescimento entre Likes e Comentários**: Uma linha de tendência foi adicionada para mostrar como o número de likes recebidos influencia o número de comentários recebidos.
- **Crescimento entre Postagens e Likes**: Uma linha de tendência foi adicionada para mostrar como o número de postagens diárias influencia o número de likes recebidos.
- **Crescimento entre Postagens e Comentários**: Uma linha de tendência foi adicionada para mostrar como o número de postagens diárias influencia o número de comentários recebidos.
""")

# Análise 8: Predição de Emoções Dominantes
st.subheader("Predição de Emoções Dominantes")
# Já feita anteriormente com o modelo carregado e mostrado nos dados com previsões

# Análise 9: Análise de Sentimentos e Uso da Plataforma
st.subheader("Análise de Sentimentos e Uso da Plataforma")
emotion_usage = data.groupby('Predicted_Emotion')['Daily_Usage_Time (minutes)'].mean().reset_index()
fig9 = px.bar(emotion_usage, x='Predicted_Emotion', y='Daily_Usage_Time (minutes)', title='Tempo Diário Médio por Emoção Dominante', color='Predicted_Emotion', color_discrete_sequence=px.colors.qualitative.Pastel, height=600)
st.plotly_chart(fig9, use_container_width=True)

# Análise 10: Análise de Diferenças de Engajamento por Emoção
st.subheader("Análise de Diferenças de Engajamento por Emoção")
col5, col6 = st.columns(2)
with col5:
    engagement_emotion = data.groupby('Predicted_Emotion')[['Likes_Received_Per_Day', 'Comments_Received_Per_Day']].mean().reset_index()
    fig10_1 = px.bar(engagement_emotion, x='Predicted_Emotion', y='Likes_Received_Per_Day', title='Likes Recebidos por Emoção Dominante', color='Predicted_Emotion', color_discrete_sequence=px.colors.qualitative.Pastel, height=600)
    st.plotly_chart(fig10_1, use_container_width=True)
with col6:
    fig10_2 = px.bar(engagement_emotion, x='Predicted_Emotion', y='Comments_Received_Per_Day', title='Comentários Recebidos por Emoção Dominante', color='Predicted_Emotion', color_discrete_sequence=px.colors.qualitative.Pastel, height=600)
    st.plotly_chart(fig10_2, use_container_width=True)

# Análise 11: Distribuição de Emoções por Faixa Etária
st.subheader("Distribuição de Emoções por Faixa Etária")
data['Age_Group'] = pd.cut(data['Age'], bins=[20, 25, 30, 35, 40], labels=['20-25', '26-30', '31-35', '36-40'])
emotion_age_group = data.groupby(['Age_Group', 'Predicted_Emotion']).size().reset_index(name='Count')
fig11 = px.bar(emotion_age_group, x='Age_Group', y='Count', color='Predicted_Emotion', barmode='stack', title='Distribuição de Emoções por Faixa Etária', color_discrete_sequence=px.colors.qualitative.Pastel, height=600)
st.plotly_chart(fig11, use_container_width=True)

# Análise 12: Análise de Interações por Tempo de Uso Diário
st.subheader("Análise de Interações por Tempo de Uso Diário")
usage_interactions = data.groupby('Daily_Usage_Time (minutes)')['Messages_Sent_Per_Day'].mean().reset_index()
fig12 = px.line(usage_interactions, x='Daily_Usage_Time (minutes)', y='Messages_Sent_Per_Day', title='Média de Mensagens Enviadas por Tempo de Uso Diário', markers=True, height=600)
st.plotly_chart(fig12, use_container_width=True)

# Formulário para entrada de dados individuais
st.subheader("Preencher Dados Individuais para Previsão de Emoção")

with st.form("prediction_form"):
    age = st.number_input("Idade", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Gênero", options=["Male", "Female"])
    platform = st.selectbox("Plataforma", options=["Instagram", "Facebook", "Twitter", "LinkedIn", "Whatsapp", "Snapchat"])
    daily_usage_time = st.number_input("Tempo Diário de Uso (minutos)", min_value=0, max_value=1440, value=60)
    posts_per_day = st.number_input("Postagens por Dia", min_value=0, max_value=100, value=1)
    likes_received_per_day = st.number_input("Likes Recebidos por Dia", min_value=0, max_value=10000, value=100)
    comments_received_per_day = st.number_input("Comentários Recebidos por Dia", min_value=0, max_value=10000, value=10)
    messages_sent_per_day = st.number_input("Mensagens Enviadas por Dia", min_value=0, max_value=10000, value=20)
    
    submitted = st.form_submit_button("Prever Emoção")

    if submitted:
        # Criar um DataFrame com os dados inseridos
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Platform': [platform],
            'Daily_Usage_Time (minutes)': [daily_usage_time],
            'Posts_Per_Day': [posts_per_day],
            'Likes_Received_Per_Day': [likes_received_per_day],
            'Comments_Received_Per_Day': [comments_received_per_day],
            'Messages_Sent_Per_Day': [messages_sent_per_day]
        })

        # Pré-processar os dados
        preprocessed_input_data = preprocess(input_data)
        
        # Prever a emoção
        predicted_emotion = opt_gb.predict(preprocessed_input_data)[0]
        
        # Mostrar o resultado
        st.write(f"A emoção dominante prevista é: {predicted_emotion}")
