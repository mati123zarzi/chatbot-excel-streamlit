import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- 1. Configuración Global (Aquí puedes cambiar el modelo) ---

# Elige tu modelo: "gemini-2.5-flash" (rápido y barato) o "gemini-2.5-pro" (más potente y caro)
MODELO_SELECCIONADO = "gemini-2.5-flash"

# Diccionario de precios en USD por 1,000,000 de tokens
PRECIOS_MODELOS = {
    "gemini-2.5-flash": {"input": 0.30, "output": 0.30}, 
    "gemini-2.5-pro": {"input": 3.50, "output": 10.50}  
}

# Selecciona los precios actuales basados en el modelo
PRECIOS_ACTUALES = PRECIOS_MODELOS.get(MODELO_SELECCIONADO, {"input": 0, "output": 0})

# --- 2. Configuración de la Página y Barra Lateral ---

st.set_page_config(page_title="Chatbot de Excel", page_icon="🤖")
st.title("🤖 Chatbot Conectado a Excel")

# Inicializar el costo total en el estado de la sesión si no existe
if "costo_total" not in st.session_state:
    st.session_state.costo_total = 0.0

# --- Barra Lateral (Sidebar) ---
st.sidebar.title("Configuración y Costos")
st.sidebar.markdown(f"**Modelo:**")
st.sidebar.code(MODELO_SELECCIONADO, language="text")

st.sidebar.markdown(f"**Precio Input:** ${PRECIOS_ACTUALES['input']:.2f} / 1M tokens")
st.sidebar.markdown(f"**Precio Output:** ${PRECIOS_ACTUALES['output']:.2f} / 1M tokens")

st.sidebar.divider()

st.sidebar.subheader("Costo Total (Realista)")
st.sidebar.metric(label="Costo Total de la Sesión", value=f"${st.session_state.costo_total:.8f} USD")

st.sidebar.divider()
st.sidebar.warning(
    "**ESTIMACIÓN REALISTA:** Este costo incluye los 'pasos intermedios' (pensamientos, código generado, etc.) "
    "y es mucho más preciso. El costo faltante (mínimo) es el 'prompt del sistema' que se envía en cada paso."
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
            allow_dangerous_code=True,
            return_intermediate_steps=True  # <-- ¡NUEVO Y CRUCIAL!
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Mostrar el desglose de costos guardado si existe
        if "cost_breakdown" in message:
            with st.expander("Ver desglose de costos de esta respuesta"):
                st.markdown(message["cost_breakdown"])

# --- 5. Lógica del Chat (con Cálculo de Costo Realista) ---
if agent and llm: 
    if prompt := st.chat_input("¿Qué quieres saber de tu Excel?"):
        
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            with st.spinner("Pensando..."):
                
                # --- CÁLCULO DE COSTO REALISTA (INICIO) ---
                total_input_tokens = 0
                total_output_tokens = 0
                
                # 1. Tokens de tu pregunta (Input)
                prompt_tokens = llm.get_num_tokens(prompt)
                total_input_tokens += prompt_tokens

                # --- Invocación del Agente ---
                response = agent.invoke(prompt)
                respuesta_agente = response['output']

                # 2. Tokens de la respuesta final (Output)
                output_tokens = llm.get_num_tokens(respuesta_agente)
                total_output_tokens += output_tokens

                # 3. Tokens de los pasos intermedios (¡El costo oculto!)
                cost_breakdown_md = f"* **Pregunta Usuario:** {prompt_tokens} tokens (input)\n"
                
               if "intermediate_steps" in response:
                    for step in response["intermediate_steps"]:
                        action = step[0]  # Es el objeto AgentAction
                        observation = str(step[1]) # Es el resultado (str)

                        # --- LA CORRECCIÓN ESTÁ AQUÍ ---
                        # Convertimos el "pensamiento" y el "código" a string INMEDIATAMENTE
                        # antes de que Langchain/Protobuf intenten leerlos.
                        action_log_str = str(action.log)               # <-- CAMBIO
                        tool_input_str = str(action.tool_input)        # <-- CAMBIO
                        # --- FIN DE LA CORRECCIÓN ---

                        # Ahora contamos los tokens de los strings, no de los objetos
                        thought_tokens = llm.get_num_tokens(action_log_str)
                        tool_input_tokens = llm.get_num_tokens(tool_input_str)
                        total_output_tokens += thought_tokens + tool_input_tokens
                        
                        # La "observación" (resultado del código) es INPUT para el LLM en el siguiente paso
                        observation_tokens = llm.get_num_tokens(observation)
                        total_input_tokens += observation_tokens

                        # Guardar para el desglose
                        cost_breakdown_md += f"* **Pensamiento/Código (Output):** {thought_tokens + tool_input_tokens} tokens\n"
                        cost_breakdown_md += f"* **Observación (Input):** {observation_tokens} tokens\n"

                cost_breakdown_md += f"* **Respuesta Final:** {output_tokens} tokens (output)\n"
                cost_breakdown_md += f"**TOTAL: {total_input_tokens} (Input) + {total_output_tokens} (Output) tokens**"
                
                # --- Cálculo de Costo Final ---
                costo_input = (total_input_tokens / 1_000_000) * PRECIOS_ACTUALES["input"]
                costo_output = (total_output_tokens / 1_000_000) * PRECIOS_ACTUALES["output"]
                costo_pregunta = costo_input + costo_output
                
                st.session_state.costo_total += costo_pregunta

            # Mostrar la respuesta del agente
            st.chat_message("assistant").markdown(respuesta_agente)
            
            # Mostrar el desglose de costos
            with st.expander("Ver desglose de costos de esta respuesta"):
                st.markdown(cost_breakdown_md)
            
            # Guardar en el historial (con el desglose)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": respuesta_agente,
                "cost_breakdown": cost_breakdown_md # Guardamos el desglose
            })
            
            st.rerun() # Recargar la UI para actualizar el costo total en la sidebar

        except Exception as e:
            st.error(f"Hubo un error al procesar tu pregunta: {e}")
else:

    st.warning("El agente no está disponible. Revisa los errores en la configuración.")


