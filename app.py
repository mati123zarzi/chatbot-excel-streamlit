import streamlit as st
import pandas as pd
import os
import io
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- 1. Configuración Global ---
MODELO_SELECCIONADO = "gemini-2.5-flash" 
TASA_CAMBIO_CLP = 950

PRECIOS_MODELOS_USD = {
    "gemini-2.5-flash": {"input": 0.30, "output": 0.30}, 
    "gemini-1.5-pro-latest": {"input": 3.50, "output": 10.50}  
}

PRECIOS_USD = PRECIOS_MODELOS_USD.get(MODELO_SELECCIONADO, {"input": 0, "output": 0})

# --- 2. Configuración de la Página y Barra Lateral ---
st.set_page_config(page_title="Chatbot de Excel", page_icon="🤖")
st.title("🤖 Chatbot Conectado a Excel")

if "costo_total" not in st.session_state:
    st.session_state.costo_total = 0.0

st.sidebar.title("Configuración y Costos")
st.sidebar.markdown(f"**Modelo:**")
st.sidebar.code(MODELO_SELECCIONADO, language="text")

precio_input_clp = PRECIOS_USD['input'] * TASA_CAMBIO_CLP
precio_output_clp = PRECIOS_USD['output'] * TASA_CAMBIO_CLP
st.sidebar.markdown(f"**Precio Input:** ${precio_input_clp:.2f} CLP / 1M tokens")
st.sidebar.markdown(f"**Precio Output:** ${precio_output_clp:.2f} CLP / 1M tokens")

st.sidebar.divider()
st.sidebar.subheader("Costo Total (Estimado)")
st.sidebar.metric(label="Costo Total de la Sesión", value=f"${st.session_state.costo_total:.4f} CLP")

st.sidebar.divider()
st.sidebar.warning(
    "**ESTIMACIÓN:** El costo real será mayor. Este cálculo solo"
    " mide tu pregunta (input) y la respuesta final (output), "
    "NO los 'pensamientos' internos del agente."
)

# --- 3. FUNCIONES DE CARGA ---

# Función 1 (Cacheada) - Carga SOLO el Excel
@st.cache_resource
def load_excel_data():
    print("Iniciando y cargando recursos cacheados (DF y Schema)...")
    
    nombre_archivo = "datos.xlsx"
    try:
        df = pd.read_excel(nombre_archivo)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{nombre_archivo}'.")
        return None, None

    print("Capturando schema del DataFrame...")
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_schema = buffer.getvalue()
    
    print("Datos de Excel cacheados listos.")
    return df, df_schema

# Función 2 (Sin Cache) - Crea el LLM y el Agente frescos
def create_fresh_agent_and_llm(df, df_schema):
    print("Creando un agente 'stateless' fresco (LLM + Agente)...")
    
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception as e:
        st.error("Error al cargar la GOOGLE_API_KEY.")
        return None, None

    try:
        llm = ChatGoogleGenerativeAI(model=MODELO_SELECCIONADO)
    except Exception as e:
        st.error(f"Error al cargar el modelo Gemini '{MODELO_SELECCIONADO}': {e}")
        return None, None

    AGENT_PREFIX = f"""
    Estás trabajando con un DataFrame de pandas en Python. El nombre del DataFrame es `df`.
    No debes modificar el DataFrame de ninguna manera (no uses inplace=True).
    Esta es la estructura (schema) completa del DataFrame con la que debes trabajar:
    <schema>
    {df_schema}
    </schema>
    Basado en el schema, responde la pregunta del usuario.
    """
    
    try:
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="openai-functions",
            verbose=True,
            allow_dangerous_code=True,
            prefix=AGENT_PREFIX
        )
        print("¡Agente y LLM frescos creados exitosamente!")
        return agent, llm
    except Exception as e:
        st.error(f"Error al crear el agente: {e}")
        return None, None

# --- 4. Inicialización del Historial de Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "cost_summary" in message:
            with st.expander("Ver detalle del costo"):
                st.markdown(message["cost_summary"])

# --- 5. Lógica del Chat ---

df, df_schema = load_excel_data()

if prompt := st.chat_input("¿Qué quieres saber de tu Excel?"):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        agent, llm = create_fresh_agent_and_llm(df, df_schema)
        
        if agent and llm: 
            with st.spinner("Pensando..."):
                
                input_tokens = llm.get_num_tokens(prompt)
                
                response = agent.invoke(prompt)
                respuesta_agente = response['output']

                # --- LÍNEA CORREGIDA ---
                output_tokens = llm.get_num_tokens(respuesta_agente)
                # -----------------------
                
                costo_input_clp = (input_tokens / 1_000_000) * PRECIOS_USD["input"] * TASA_CAMBIO_CLP
                costo_output_clp = (output_tokens / 1_000_000) * PRECIOS_USD["output"] * TASA_CAMBIO_CLP
                costo_pregunta_clp = costo_input_clp + costo_output_clp
                
                st.session_state.costo_total += costo_pregunta_clp

                cost_summary = f"""
                * **Coste de esta pregunta (Estimado):** `${costo_pregunta_clp:.8f} CLP`
                * **Tokens Entrada:** `{input_tokens}` (Coste: `${costo_input_clp:.8f} CLP`)
                * **Tokens Salida:** `{output_tokens}` (Coste: `${costo_output_clp:.8f} CLP`)
                """

            st.chat_message("assistant").markdown(respuesta_agente)
            
            with st.expander("Ver detalle del costo"):
                st.markdown(cost_summary)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": respuesta_agente,
                "cost_summary": cost_summary
            })
            
            st.rerun()

        else:
            st.error("No se pudo inicializar el agente. Revisa la configuración.")

    except Exception as e:
        st.error(f"Hubo un error al procesar tu pregunta: {e}")


