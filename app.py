import streamlit as st
import pandas as pd
import os
import io
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- 1. Configuración Global ---
MODELO_SELECCIONADO = "gemini-2.5-flash" 
TASA_CAMBIO_CLP = 950  # <-- NUEVA TASA DE CAMBIO

# Precios base en USD por 1,000,000 de tokens
PRECIOS_MODELOS_USD = {
    "gemini-2.5-flash": {"input": 0.30, "output": 0.30}, 
    "gemini-1.5-pro-latest": {"input": 3.50, "output": 10.50}  
}

# Selecciona los precios actuales en USD
PRECIOS_USD = PRECIOS_MODELOS_USD.get(MODELO_SELECCIONADO, {"input": 0, "output": 0})

# --- 2. Configuración de la Página y Barra Lateral ---
st.set_page_config(page_title="Chatbot de Excel", page_icon="🤖")
st.title("🤖 Chatbot Conectado a Excel")

if "costo_total" not in st.session_state:
    st.session_state.costo_total = 0.0 # El costo total se acumulará en CLP

st.sidebar.title("Configuración y Costos")
st.sidebar.markdown(f"**Modelo:**")
st.sidebar.code(MODELO_SELECCIONADO, language="text")

# --- CAMBIO A CLP: Mostrar precios en CLP ---
precio_input_clp = PRECIOS_USD['input'] * TASA_CAMBIO_CLP
precio_output_clp = PRECIOS_USD['output'] * TASA_CAMBIO_CLP
st.sidebar.markdown(f"**Precio Input:** ${precio_input_clp:.2f} CLP / 1M tokens")
st.sidebar.markdown(f"**Precio Output:** ${precio_output_clp:.2f} CLP / 1M tokens")

st.sidebar.divider()
st.sidebar.subheader("Costo Total (Estimado)")
# --- CAMBIO A CLP: Mostrar costo total en CLP ---
st.sidebar.metric(label="Costo Total de la Sesión", value=f"${st.session_state.costo_total:.4f} CLP")

st.sidebar.divider()
st.sidebar.warning(
    "**ESTIMACIÓN:** El costo real será mayor. Este cálculo solo"
    " mide tu pregunta (input) y la respuesta final (output), "
    "NO los 'pensamientos' internos del agente."
)

# --- 3. Función para Cargar el Agente (Modo Stateless) ---
def get_agent_and_llm():
    print("Iniciando y cargando el agente (Modo Stateless)...")
    
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception as e:
        st.error("Error al cargar la GOOGLE_API_KEY. ¿La agregaste en los 'Secrets' de Streamlit?")
        return None, None

    try:
        llm = ChatGoogleGenerativeAI(model=MODELO_SELECCIONADO)
    except Exception as e:
        st.error(f"Error al cargar el modelo Gemini '{MODELO_SELECCIONADO}': {e}")
        return None, None

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
        print("¡Agente cargado exitosamente con schema!")
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
if prompt := st.chat_input("¿Qué quieres saber de tu Excel?"):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Creamos un agente FRESCO para CADA pregunta
        agent, llm = get_agent_and_llm()
        
        if agent and llm: 
            with st.spinner("Pensando..."):
                
                input_tokens = llm.get_num_tokens(prompt)
                
                response = agent.invoke(prompt)
                respuesta_agente = response['output']

                output_tokens = llm.get_num_tokens(respuesta_agente)
                
                # --- CAMBIO A CLP: Calcular costos en CLP ---
                costo_input_clp = (input_tokens / 1_000_000) * PRECIOS_USD["input"] * TASA_CAMBIO_CLP
                costo_output_clp = (output_tokens / 1_000_000) * PRECIOS_USD["output"] * TASA_CAMBIO_CLP
                costo_pregunta_clp = costo_input_clp + costo_output_clp
                
                # Acumulamos el costo en CLP
                st.session_state.costo_total += costo_pregunta_clp

                # --- CAMBIO A CLP: Mostrar resumen en CLP ---
                # Usamos más decimales para ver costos muy pequeños
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
