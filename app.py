import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- 1. Configuración Global (Aquí puedes cambiar el modelo) ---
MODELO_SELECCIONADO = "gemini-2.5-flash" 

PRECIOS_MODELOS = {
    "gemini-2.5-flash": {"input": 0.30*950, "output": 0.30*950}, 
    "gemini-2.5-pro-latest": {"input": 3.50*950, "output": 10.50*950}  
}

PRECIOS_ACTUALES = PRECIOS_MODELOS.get(MODELO_SELECCIONADO, {"input": 0, "output": 0})

# --- 2. Configuración de la Página y Barra Lateral ---
st.set_page_config(page_title="Chatbot de Excel", page_icon="🤖")
st.title("🤖 Chatbot Conectado a Excel")

if "costo_total" not in st.session_state:
    st.session_state.costo_total = 0.0

st.sidebar.title("Configuración y Costos")
st.sidebar.markdown(f"**Modelo:**")
st.sidebar.code(MODELO_SELECCIONADO, language="text")
st.sidebar.markdown(f"**Precio Input:** ${PRECIOS_ACTUALES['input']:.2f} / 1M tokens")
st.sidebar.markdown(f"**Precio Output:** ${PRECIOS_ACTUALES['output']:.2f} / 1M tokens")
st.sidebar.divider()
st.sidebar.subheader("Costo Total (Estimado)")
st.sidebar.metric(label="Costo Total de la Sesión", value=f"${st.session_state.costo_total:.8f} CLP")
st.sidebar.divider()
st.sidebar.warning(
    "**ESTIMACIÓN:** El costo real será mayor. Este cálculo solo"
    " mide tu pregunta (input) y la respuesta final (output), "
    "NO los 'pensamientos' internos del agente."
)

# --- 3. Función para Cargar el Agente y el LLM (Cacheada) ---
@st.cache_resource
def get_agent_and_llm():
    print("Iniciando y cargando el agente...")
    
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

    try:
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="openai-functions",
            verbose=True,
            allow_dangerous_code=True 
        )
        print("¡Agente cargado exitosamente!")
        return agent, llm
    except Exception as e:
        st.error(f"Error al crear el agente: {e}")
        return None, None

# --- 4. Inicialización del Chat ---
agent, llm = get_agent_and_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CAMBIO: Mostrar el detalle de costo si existe en el historial ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Mostrar el desglose de costos guardado si existe
        if "cost_summary" in message:
            with st.expander("Ver detalle del costo"):
                st.markdown(message["cost_summary"])

# --- 5. Lógica del Chat (con Cálculo de Costo Simplificado) ---
if agent and llm: 
    if prompt := st.chat_input("¿Qué quieres saber de tu Excel?"):
        
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            with st.spinner("Pensando..."):
                
                # --- CÁLCULO DE COSTO SIMPLIFICADO (Input) ---
                input_tokens = llm.get_num_tokens(prompt)
                
                # --- Invocación del Agente (simple) ---
                response = agent.invoke(prompt)
                respuesta_agente = response['output']

                # --- CÁLCULO DE COSTO SIMPLIFICADO (Output y Total) ---
                output_tokens = llm.get_num_tokens(respuesta_agente)
                
                costo_input = (input_tokens / 1_000_000) * PRECIOS_ACTUALES["input"]
                costo_output = (output_tokens / 1_000_000) * PRECIOS_ACTUALES["output"]
                costo_pregunta = costo_input + costo_output
                
                st.session_state.costo_total += costo_pregunta

                # --- CAMBIO: Crear el texto para el expander ---
                cost_summary = f"""
                * **Coste de esta pregunta (Estimado):** `${costo_pregunta:.8f} CLP`
                * **Tokens Entrada:** `{input_tokens}` (Coste: `${costo_input:.8f}`)
                * **Tokens Salida:** `{output_tokens}` (Coste: `${costo_output:.8f}`)
                """

            # Mostrar la respuesta del agente
            st.chat_message("assistant").markdown(respuesta_agente)
            
            # --- CAMBIO: Mostrar el expander en el chat ---
            with st.expander("Ver detalle del costo"):
                st.markdown(cost_summary)
            
            # --- CAMBIO: Guardar el detalle en el historial ---
            st.session_state.messages.append({
                "role": "assistant", 
                "content": respuesta_agente,
                "cost_summary": cost_summary # Guardamos el detalle
            })
            
            st.rerun() # Recargar la UI para actualizar el costo total en la sidebar

        except Exception as e:
            st.error(f"Hubo un error al procesar tu pregunta: {e}")
else:
    st.warning("El agente no está disponible. Revisa los errores en la configuración.")

