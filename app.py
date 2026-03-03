# ESPECIALIZACION PYTHON FOR ANALYTICS
# TRABAJO FINAL
# AUTOR: LUZ IBAÑEZ

# IMPORTACION DE LIBRERIAS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# ESTILOS HTML + CSS
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* FULL WIDTH con márgenes laterales */
section.main > div.block-container {
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 100% !important;
    width: 100% !important;
}

/* Compatibilidad con versiones anteriores */
div.block-container {
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 100% !important;
}

/* Variables automáticas según modo claro/oscuro */
:root {
    --card-bg: var(--secondary-background-color);
    --text-color: var(--text-color);
}

/* Títulos */
h1 {
    color: var(--text-color);
    font-weight: 700;
    text-align: center;
    padding-bottom: 10px;
}

/* Subtítulo centrado */
.subtitle-center {
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-color);
    margin-top: -10px;
    margin-bottom: 20px;
}

/* Tarjetas con borde azul */
.card {
    background: var(--card-bg);
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 25px;
    border: 2px solid #2196f3;   /* AZUL */
    width: 100%;
    box-sizing: border-box;
}

/* Listas con check */
ul.checklist li::marker {
    content: "✔️ ";
}

</style>
""", unsafe_allow_html=True)

# CREACION DE MENU
st.sidebar.title("🙋🙋‍♂️ Bienvenido(a)")
interfaz = st.sidebar.selectbox(
    "Elija una opción:",
    [
        "Home 🏡",
        "Desarrollo EDA 📈",
        "Conclusiones Finales 📌"
    ]
)

# DESARROLLO DE HOME
if interfaz == "Home 🏡":

    st.title("Proyecto Final de Especialización de Python for Analytics")
    st.write("---")
    st.image(
        "imagen_home.jpg",
        use_container_width=True
    )

    st.subheader("Caso de Estudio N°2: Fuga de clientes")

    st.markdown("""
    <div class="card">
        <h3>🎯 Objetivo del análisis</h3>
        <p>Analizar los datos de clientes e identificar las causas de la fuga presentada.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>👩‍💻 Datos del autor</h3>
        <p><b>Nombre:</b> Luz de Maria Ibañez Berrospi<br>
        <b>Curso/Especialización:</b> Python for Analytics<br>
        <b>Año:</b> 2026</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>📊 Dataset TelcoCustomerChurn</h3>
        <p>Contiene información sobre los clientes, sus servicios contratados, facturación mensual, tiempo de permanencia y estado actual en la empresa.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🛠️ Tecnologías utilizadas</h3>
        <ul class="checklist">
            <li>Python</li>
            <li>Streamlit</li>
            <li>Pandas</li>
            <li>Numpy</li>
            <li>Matplotlib</li>
            <li>Seaborn</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# DESARROLLO DE ITEMS
elif interfaz == "Desarrollo EDA 📈":
    # CREACION DE CLASE
    class Data_Processor:
        #CREACION DE CONSTRUCTOR
        def __init__(self, df):
            self.df = df

        #CREACION DE FUNCION INFORMACION
        def informacion(self):
            df_tipo_dato = pd.DataFrame({
                "Campo": self.df.columns,
                "Tipo de dato": self.df.dtypes.values
            })
            return df_tipo_dato

        #CREACION DE FUNCION CANTIDAD_VALORES_NULOS
        def cantidad_valores_nulos(self):
            df_valores_nulos = pd.DataFrame({
                "Campo": self.df.columns,
                "Cantidad de valores nulos": self.df.isnull().sum().values
            })
            return df_valores_nulos

        #CREACION DE FUNCION IDENTIFICACION DE TIPO DE VARIABLE
        def identificacion_tipo_variable(self):
            var_categoricas = self.df.select_dtypes(include=["string"]).columns.tolist()
            var_numericas = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()  
            return var_numericas, var_categoricas

        # CREACION DE FUNCION QUE DEVUELVE ESTADISTICAS DESCRIPTIVAS DEL DATASET
        def estadisticas_descriptivas(self):
            return self.df.describe()

        #CREACION DE FUNCION QUE DEVUELVE LA MEDIA
        def media(self, col):
            return self.df[col].mean()
        
        #CREACION DE FUNCION QUE DEVUELVE LA MODA
        def moda(self, col):
            return self.df[col].mode()[0]
        
        #CREACION DE FUNCION QUE DEVUELVE LA MEDIANA
        def mediana(self, col):
            return self.df[col].median()
        
        #CREACION DE FUNCION QUE DEVUELVE LA VARIANZA
        def varianza(self, col):
            # ddof=1 para varianza muestral
            return self.df[col].var(ddof=1)

        #CREACION DE FUNCION QUE DEVUELVE LA DESVIACIÓN ESTANDAR
        def desviacion_estandar(self, col):
            return self.df[col].std(ddof=1)

        #CREACION DE FUNCION QUE DEVUELVE EL RANGO INTERCUALITICO
        def iqr(self, col):
            return self.df[col].quantile(0.75) - self.df[col].quantile(0.25)

    st.title("Análisis Exploratorio de Datos (EDA) 📈")
    st.write("---")
    st.subheader("Carga de dataset")
    archivo = st.file_uploader("A continuación, cargue un archivo tipo .CSV:", type=["csv"])

    if archivo is not None:
        try:
            #LECTURA DE ARCHIVO .CSV
            df = pd.read_csv(archivo)

            #CONVERSIÓN DE CAMPOS DE TIPO OBJECT
            df["customerID"] = df["customerID"].astype("string")
            df["gender"] = df["gender"].astype("string")
            df["SeniorCitizen"] = df["SeniorCitizen"].astype("string")
            df["Partner"] = df["Partner"].astype("string")
            df["Dependents"] = df["Dependents"].astype("string")
            df["PhoneService"] = df["PhoneService"].astype("string")
            df["MultipleLines"] = df["MultipleLines"].astype("string")
            df["InternetService"] = df["InternetService"].astype("string")
            df["OnlineSecurity"] = df["OnlineSecurity"].astype("string")
            df["OnlineBackup"] = df["OnlineBackup"].astype("string")
            df["DeviceProtection"] = df["DeviceProtection"].astype("string")
            df["TechSupport"] = df["TechSupport"].astype("string")
            df["StreamingTV"] = df["StreamingTV"].astype("string")
            df["StreamingMovies"] = df["StreamingMovies"].astype("string")
            df["Contract"] = df["Contract"].astype("string")
            df["PaperlessBilling"] = df["PaperlessBilling"].astype("string")
            df["PaymentMethod"] = df["PaymentMethod"].astype("string")
            df["Churn"] = df["Churn"].astype("string")
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
      
            #CREACION DE OBJETO
            objeto = Data_Processor(df)

            #MENSAJE DE CARGA EXITOSA
            st.success("El archivo se cargó con éxito")

            #VISTA DE LAS PRIMERAS 5 FILAS DEL DATASET
            st.subheader("Vista previa (5 primeras filas del dataset)")
            st.dataframe(df.head(6).reset_index(drop=True))

            #CANTIDAD DE FILAS Y COLUMNAS DEL DATASET
            cant_filas, cant_columnas = df.shape
            st.subheader("Dimensiones")
            st.write(f"El dataset consta de: **{cant_filas} filas y {cant_columnas} columnas**.")

            # ITEMS           
            tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs(["Ítem 1️⃣", "Ítem 2️⃣", "Ítem 3️⃣","Ítem 4️⃣","Ítem 5️⃣","Ítem 6️⃣","Ítem 7️⃣","Ítem 8️⃣","Ítem 9️⃣","Ítem 🔟"])

            # DESARROLLO DE ITEM 1
            with tab1:
                st.title("Ítem 1️⃣: Información general del dataset")
                #MUESTRA INFORMACIÓN GENERAL DEL DATASET
                st.dataframe(objeto.informacion())

                st.subheader("Conteo de valores nulos")
                #MUESTRA INFORMACIÓN DE VALORES NULOS ENCONTRADOS EN EL DATASET
                st.dataframe(objeto.cantidad_valores_nulos().sort_values(by="Cantidad de valores nulos", ascending=False))

            # DESARROLLO DE ITEM 2
            with tab2:
                st.title("Ítem 2️⃣: Clasificación de variables")
                var_numericas, var_categoricas = objeto.identificacion_tipo_variable()
                
                #CANTIDAD Y LISTADO DE VARIABLES NUMÉRICAS
                st.write(f"Existen **{len(var_numericas)}** variables numéricas:")
                for col in var_numericas:
                    st.write("✔️", col)
                
                #CANTIDAD Y LISTADO DE VARIABLES CATEGÓRICAS
                st.write(f"Y **{len(var_categoricas)}** variables categóricas:")
                cols = st.columns(6)

                # COLOCAR EN 6 COLUMNAS LAS VARIABLES CATEGÓRICAS
                for i, col in enumerate(var_categoricas):
                    cols[i % 6].write("✔️ " + col)


            # DESARROLLO DE ITEM 3
            with tab3:
                st.title("Ítem 3️⃣: Estadísticas descriptivas")

                #SE OBTIENE LA INFORMACIÓN .DESCRIBE()
                st.dataframe(objeto.estadisticas_descriptivas())

                #CALCULO DE MEDIDAS
                st.subheader("Cálculo de medidas")
                #SOLO SE RECEPCIONA LA LISTA DE VALORES NUMÉRICOS
                numericas, _ = objeto.identificacion_tipo_variable()

                #SEGÚN LA VARIABLE ELEGIDA, SE CALCULA: MEDIA, MODA Y MEDIANA
                opcion = st.selectbox(
                    "Variable Numérica:",
                    ["---Selecciona---"] + numericas,
                     index=0,
                     key="select_box_3"
                )

                #INTERPRETACIÓN DE LAS MEDIDAS:
                st.write("**INTERPRETACIÓN:**")
                if opcion != "---Selecciona---":
                    if opcion=="tenure":
                        st.write(f"✔️ Media: En promedio, los clientes permanecen **{objeto.media(opcion):.2f}** mes(es) en la empresa hasta dar de baja a los servicios contratados.")                   
                        st.write(f"✔️ Moda: Comúnmente, los clientes suelen dar de baja a los servicios contratados luego de **{objeto.moda(opcion)}** mes(es).")
                        st.write(f"✔️ Mediana: La mitad los clientes se va antes de cumplir **{objeto.mediana(opcion):.0f}** mes(es), y la otra mitad permanece más allá de ese tiempo.")
                        st.write(f"✔️ Varianza: La varianza obtenida fue **{objeto.varianza(opcion):.2f}**, lo que indica una dispersión relativamente alta de la cantidad de meses de permanencia de los clientes.")
                        st.write(f"✔️ Desviación estándar: La desviación estándar obtenida fue de **{objeto.desviacion_estandar(opcion):.2f}** meses de la media de permanencia, lo que significa una alta variabilidad en el tiempo que los clientes permanecen en la empresa.")                    
                        st.write(f"✔️ IQR: La mitad de los clientes presenta tiempos de permanencia que varían en un intervalo de **{objeto.iqr(opcion):.2f}** meses.")
                    
                    elif opcion=="MonthlyCharges":
                        st.write(f"✔️ Media: El cargo mensual promedio de los clientes es **S/.{objeto.media(opcion):.2f}.**")                   
                        st.write(f"✔️ Moda: Comúnmente, el cargo mensual de los clientes es **S/.{objeto.moda(opcion):.2f}**.")
                        st.write(f"✔️ Mediana: La mitad de los clientes tienen un cargo mensual menor a **S/.{objeto.mediana(opcion):.2f}** y la otra mitad tienen un cargo mayor a ese.")
                        st.write(f"✔️ Varianza: La varianza obtenida fue **{objeto.varianza(opcion):.2f}**, lo que significa indica una dispersión significativa en los cargos mensuales de los clientes.")
                        st.write(f"✔️ Desviación estándar: Los cargos mensuales de los clientes se alejan aproximandamente S/.**{objeto.desviacion_estandar(opcion):.2f}** de la media, lo que indica una alta variabilidad en los cargos cobrados a los clientes.")                       
                        st.write(f"✔️ IQR: La mitad de los clientes presentan cargos mensuales que varían dentro de un intervalo de S/.**{objeto.iqr(opcion):.2f}**.")
                    

                    elif opcion=="TotalCharges":
                        st.write(f"✔️ Media: El cargo total promedio de los clientes es **S/.{objeto.media(opcion):.2f}.**")                   
                        st.write(f"✔️ Moda: Comúnmente, el cargo total de los clientes es **S/.{objeto.moda(opcion):.2f}**.")
                        st.write(f"✔️ Mediana: La mitad de los clientes tienen un cargo total menor a **S/.{objeto.mediana(opcion):.2f}** y la otra mitad tienen un cargo mayor a ese.")
                        st.write(f"✔️ Varianza: La varianza obtenida fue **{objeto.varianza(opcion):.2f}**,  ha cobrado a los clientes.")
                        st.write(f"✔️ Desviación estándar: Los cargos totales de los clientes se alejan aproximadamente S/.**{objeto.desviacion_estandar(opcion):.2f}** de la media, lo que indica una variabilidad considerable en los cargos cobrados a los clientes.")           
                        st.write(f"✔️ IQR: La mitad de los clientes presentan cargos totales que varían dentro de un intervalo de S/.**{objeto.iqr(opcion):.2f}**.")
                    

            # DESARROLLO DE ITEM 4
            with tab4:
                st.title("Ítem 4️⃣: Análisis de valores faltantes")

                # Usamos tu función para obtener el DataFrame de valores nulos
                df_valores_nulos = objeto.cantidad_valores_nulos()

                #SE CREAN LAS COLUMNAS
                col1, col2 = st.columns(2)

                with col1:
                    # CREACIÓN DE FIGURA
                    fig, ax = plt.subplots(figsize=(8,5))
                    df_valores_nulos.set_index("Campo")["Cantidad de valores nulos"].plot(kind="barh", ax=ax, color="#8E9717")
                    ax.set_xlabel("Cantidad de valores nulos")
                    ax.set_ylabel(None)
                    ax.set_title("Distribución de valores nulos por variable", fontweight="bold")

                    # AGREGAR ETIQUETAS EN LAS BARRAS
                    for i, v in enumerate(df_valores_nulos["Cantidad de valores nulos"]):
                        ax.text(v + 0.1, i, str(v), va='center')
                    st.pyplot(fig)

                with col2:
                    # OBTENCIÓN DE SOLO LAS VARIABLES CON VALORES NULOS
                    variables_con_valores_nulos = df_valores_nulos[
                        df_valores_nulos["Cantidad de valores nulos"] > 0
                    ]["Campo"].tolist()
                    
                    # OBTENCIÓN DE LA CANTIDAD DE VALORES NULOS
                    variables_con_valores_nulos = df_valores_nulos[
                        df_valores_nulos["Cantidad de valores nulos"] > 0
                    ]["Cantidad de valores nulos"].tolist()

                    # SE ANALIZA SI ES 1 O VARIAS VARIABLES CON VALORES NULOS
                    if variables_con_valores_nulos:
                        #INTERPETACIÓN
                        st.write("**INTERPRETACIÓN:**")
                        st.write(f"Variable con valores nulos: **{variables_con_valores_nulos[0]}**.")
                        st.write(f"✔️ Significa que existen **{variables_con_valores_nulos[0]}** clientes que no presentan cargo total, debido a que aún no cumplen 1 mes de permanencia."
                        )
                    else:
                        st.write("No existen variables con valores nulos.")

            # DESARROLLO DE ITEM 5
            with tab5:
                st.title("Ítem 5️⃣: Distribución de variables numéricas")
                #OBTENER LOS NOMBRES DE LAS VARIABLES NUMÉRICAS
                numericas, _ = objeto.identificacion_tipo_variable()
                opcion_item_5 = st.selectbox("Elija una variable para generar su histograma:", numericas)
                
                #CREAR LAS COLUMNAS
                col1, col2 = st.columns(2)   
                if opcion_item_5=="tenure": 
                    with col1:
                        # CREACIÓN DE FIGURA
                        fig, ax = plt.subplots()
                        sns.histplot(df[opcion_item_5], kde=True, ax=ax)

                        # AGREGAR TÍTULO AL GRÁFICO
                        ax.set_title(f"Distribución de la cantidad de meses de permanencia", fontweight="bold")

                        # CREAR DEGRADADO DE COLORES EN BARRAS
                        cmap = cm.get_cmap("viridis", len(ax.patches))

                        for i, p in enumerate(ax.patches):
                            #ASIGNAR COLOR DISTINTO A CADA BARRA
                            color = cmap(i) 
                            p.set_facecolor(color)

                            # COLOCAR ETIQUETAR A CADA BARRA
                            altura = p.get_height()
                            if altura > 0:
                                ax.text(
                                    p.get_x() + p.get_width() / 2,
                                    altura,
                                    int(altura),
                                    ha="center", va="bottom",
                                    fontsize=7, fontweight="bold"
                                )
                        st.pyplot(fig)

                    with col2:
                       #INTERPRETACIÓN
                        st.write("**INTERPRETACIÓN:**")
                        st.write("✔️ Existe una gran cantidad de clientes (1238) cuyo tiempo de permanencia varía entre 0 a 5 meses, es decir, clientes que recién han contratado servicios con la empresa.")
                        st.write("⚠️ La cantidad de clientes cuya permanencia es entre 5 y 65 meses es considerablemente muy baja con respecto al anterior punto, pero estable.")
                        st.write("✔️ Finalmente, la cantidad de clientes con un tiempo de permanencia mayor a 65 meses se eleva respecto al punto anterior, lo que refleja fidelidad por parte de los clientes antiguos.")
                
                if opcion_item_5=="MonthlyCharges": 
                    with col1:
                        # CREACIÓN DE FIGURA
                        fig, ax = plt.subplots()
                        sns.histplot(df[opcion_item_5], kde=True, ax=ax)

                        # AGREGAR TÍTULO AL GRÁFICO
                        ax.set_title(f"Distribución del cargo mensual cobrado al cliente", fontweight="bold")

                        # CREAR DEGRADADO DE COLORES EN BARRAS
                        cmap = cm.get_cmap("viridis", len(ax.patches))

                        for i, p in enumerate(ax.patches):
                            #ASIGNAR COLOR DISTINTO A CADA BARRA
                            color = cmap(i) 
                            p.set_facecolor(color)

                            # COLOCAR ETIQUETAR A CADA BARRA
                            altura = p.get_height()
                            if altura > 0:
                                ax.text(
                                    p.get_x() + p.get_width() / 2,
                                    altura,
                                    int(altura),
                                    ha="center", va="bottom",
                                    fontsize=7, fontweight="bold"
                                )
                        st.pyplot(fig)

                    with col2:
                       #INTERPRETACIÓN
                        st.write("**INTERPRETACIÓN:**")
                        st.write("✔️ La mayor cantidad de clientes (1200) pagan cargos mensuales en un rango de S/.18 a S/.25.")
                        st.write("⚠️ Se presenta una disminución de cantidad de clientes que pagan cargos mensuales entre: 30-40, 60-65 y 110-120 soles.")
                        st.write("✔️ Se observa baja variabilidad en la cantidad de los clientes que pagan cargos mensuales entre 70 a 100 soles.")
                
                if opcion_item_5=="TotalCharges": 
                    with col1:
                        # CREACIÓN DE FIGURA
                        fig, ax = plt.subplots()
                        sns.histplot(df[opcion_item_5], kde=True, ax=ax)

                        # AGREGAR TÍTULO AL GRÁFICO
                        ax.set_title(f"Distribución del cargo total cobrado al cliente", fontweight="bold")

                        # CREAR DEGRADADO DE COLORES EN BARRAS
                        cmap = cm.get_cmap("viridis", len(ax.patches))

                        for i, p in enumerate(ax.patches):
                            #ASIGNAR COLOR DISTINTO A CADA BARRA
                            color = cmap(i) 
                            p.set_facecolor(color)

                            # COLOCAR ETIQUETAR A CADA BARRA
                            altura = p.get_height()
                            if altura > 0:
                                ax.text(
                                    p.get_x() + p.get_width() / 2,
                                    altura,
                                    int(altura),
                                    ha="center", va="bottom",
                                    fontsize=7, fontweight="bold"
                                )
                        st.pyplot(fig)

                    with col2:
                        #INTERPRETACIÓN
                        st.write("**INTERPRETACIÓN:**")
                        st.write("⚠️ La mayor cantidad de clientes (1678) poseen un cargo total hasta de S/500, lo que indica que están generando un bajo ingreso a la empresa.")
                        st.write("⚠️ A medida que el cargo total cobrado aumenta, la cantidad de clientes a quienes corresponden disminuye considerablemente, lo que significa que son pocos clientes los que generan un mayor ingreso a la empresa.")
                       
            # DESARROLLO DE ITEM 6
            with tab6:
                st.title("Ítem 6️⃣: Análisis de variables categóricas")

                # OBTENER LOS NOMBRES DE LAS VARIABLES CATEGÓRICAS
                _, categoricas = objeto.identificacion_tipo_variable()
                #NO CONSIDERAR VARIABLE CUSTOMER ID, YA QUE NO ES VÁLIDO EVALUARLO INDEPENDIENTEMENTE
                categoricas = [c for c in categoricas if c != "customerID"]

                st.write("Elija 1 o más variables:")

                # CREAR 6 COLUMNAS PARA LOS CHECKBOX
                cols = st.columns(6)
                checkbox = []
                for i, col in enumerate(categoricas):
                    with cols[i % 6]:
                        if st.checkbox(f"{col}"):
                            checkbox.append(col)
                st.subheader("CONTEO")
                # GENERAR UN GRÁFICO Y SU TEXTO POR CADA VARIABLE SELECCIONADA
                for col in checkbox:
                    #CREAR 2 COLUMNAS: PARA GRÁFICO Y TEXTO
                    col7, col8 = st.columns(2)

                    with col7:
                        fig, ax = plt.subplots()
                        counts = df[col].value_counts()
                        total = counts.sum()

                        # GRÁFICO DE BARRAS CON ETIQUETA PORCENTAJE
                        counts.plot(kind="bar", ax=ax, color="#179784", edgecolor="black")
                        ax.set_title(f"Distribución de la variable {col}", fontweight="bold")
                        ax.set_ylabel("Frecuencia")
                        ax.set_xlabel(None)

                        # ETIQUETAS SOBRE LAS BARRAS
                        for p in ax.patches:
                            altura = p.get_height()
                            porcentaje = (altura / total) * 100
                            ax.text(
                                p.get_x() + p.get_width() / 2,
                                altura,
                                f"{porcentaje:.2f}%",
                                ha="center", va="bottom",
                                fontsize=8, fontweight="bold"
                            )
                        st.pyplot(fig)

                    with col8:
                        #MOSTRAR INFORMACIÓN DE LA VARIABLE
                        st.write(f"**Variable: {col}**")
                        st.write("Valores:")
                        for categoria, valor in counts.items():
                            porcentaje = (valor / total) * 100
                            st.write(f"**🔹{categoria}:** {valor} ({porcentaje:.2f}%)")
                    st.write("---")

            # DESARROLLO DE ITEM 7
            with tab7:
                st.title("Ítem 7️⃣: Análisis bivariado (numérico vs categórico)")                
                
                st.subheader("Tenure (var numérica) vs Churn (var categórica)")
                st.write("🔹 Tenure = Cantidad de meses de permanencia del cliente en la empresa")
                st.write("🔹 Churn = Si el cliente abandonó la empresa (Yes/No)")
                
                # OBTENCION DE VALOR MAXIMO Y MINIMO DE VARIABLE TENURE PARA COLOCARLO EN SLIDER
                min_tenure = int(df["tenure"].min())
                max_tenure = int(df["tenure"].max())

                # OBTENCION DEL VALOR COLOCADO EN SLIDER
                selected_tenure = st.slider(
                    f"Seleccione el máximo tiempo de permanencia (en meses) del cliente en la empresa (de {min_tenure} a {max_tenure}):",
                    min_value=min_tenure,
                    max_value=max_tenure,
                    value=max_tenure // 2,
                    step=1
                )

                #OBTENCION DE DATOS QUE CUMPLEN CON EL VALOR SELECCIONADO EN SLIDER
                filtered_df = df[df["tenure"] <= selected_tenure]

                # CALCULO DE TASA DE VARIABLE CHURN
                tasa_churn = filtered_df.groupby("tenure")["Churn"].apply(lambda x: (x == "Yes").mean())

                # CREACION DE COLUMNAS
                col1, col2 = st.columns(2)

                with col1:
                    # CREACIÓN DE GRÁFICO
                    fig, ax = plt.subplots(figsize=(10,6))
                    tasa_churn.plot(ax=ax, marker="o", color="red")
                    ax.set_title(f"Proporción de clientes que abandonaron la empresa en un tiempo ≤ {selected_tenure} meses", fontweight="bold")
                    ax.set_ylabel("Proporción de clientes que abandonaron la empresa")
                    ax.set_xlabel("Tiempo de permanencia (meses)")
                    st.pyplot(fig)

                with col2:         
                    churn_rate = (filtered_df["Churn"] == "Yes").mean().round(2) * 100
                    # INTERPRETACIÓN
                    st.write("**INTERPRETACIÓN:**")
                    st.write(f"ℹ️ Aproximadamente el {churn_rate:.2f}% de los clientes que tenían un tiempo de permanencia ≤ {selected_tenure} abandonaron la empresa.")
                    st.write("✔️ A medida que pasan los meses, el porcentaje de clientes que abandonan la empresa se reduce.")
                
                st.write("---")

                st.subheader("MonthlyCharges (var numérica) vs Contract (var categórica)")
                st.write("🔹 MonthlyCharges = Cargo mensual cobrado al cliente")
                st.write("🔹 Contract = Tipo de contrato")
                # CREAR 2 COLUMNAS PARA MOSTRAR GRÁFICO Y TEXTO
                col3,col4 = st.columns(2)
                with col3:
                    #CREACIÓN DE GRÁFICO
                    fig, ax = plt.subplots()
                    ax.set_title("Promedio de cargos mensuales por tipo de contrato", fontweight="bold")
                    sns.barplot(x="Contract", y="MonthlyCharges", data=df, ax=ax, ci=None, palette="muted")
                    ax.set_xlabel("Tipo de contrato")
                    ax.set_ylabel("Cargo mensual promedio")

                    for p in ax.patches:
                        ax.text(
                            p.get_x() + p.get_width() / 2,   
                            p.get_height(),                  
                            f'{p.get_height():.2f}',      
                            ha='center', va='bottom', fontsize=9
                        )
                    st.pyplot(fig)
                with col4:
                    #INTERPRETACIÓN
                    st.write("**INTERPRETACIÓN:**")
                    st.write("ℹ️ El cargo mensual promedio que pagan los clientes que tienen contratos mes a mes (66.40) es mayor que el promedio de cargo mensual de los contratos por 2 años (60.77), " \
                    "lo cual refleja la aplicación de una estrategia de fidelización: cargos mensuales menores en contratos de mayor tiempo.")
                    
            # DESARROLLO DE ITEM 8
            with tab8:
                st.title("Ítem 8️⃣: Análisis bivariado (categórico vs categórico)")
                st.subheader("Contract (var categórica) vs Churn (var categórica)")
                st.write("🔹 Contract = Tipo de contrato")
                st.write("🔹 Churn = Si el cliente abandonó la empresa (Yes/No)")
                # CREAR 2 COLUMNAS PARA MOSTRAR GRÁFICO Y TEXTO
                col1,col2 = st.columns(2)
                with col1:                
                    fig, ax = plt.subplots()
                    ax.set_title("Fuga de clientes por Tipo de Contrato", fontweight="bold")

                    # CREACIÓN DE GRÁFICO
                    sns.countplot(x="Contract", hue="Churn", data=df, ax=ax, palette="muted")
                    ax.set_xlabel("Tipo de contrato")
                    ax.set_ylabel("Número de clientes")
                    ax.legend(title="Cliente abandonó la empresa:")

                    for p in ax.patches:
                        height = p.get_height()
                        if height > 0:  
                            ax.text(
                                p.get_x() + p.get_width() / 2, 
                                height,                         
                                f'{int(height)}',               
                                ha='center', va='bottom', fontsize=9
                            )
                    st.pyplot(fig)

                with col2:
                    #INTERPRETACIÓN
                    st.write("**INTERPRETACIÓN:**")
                    st.write("⚠️ La mayor cantidad de clientes que abandonaron la empresa fueron aquellos que tenían un tipo de contrato Mes a Mes (1655).")
                    st.write("✔️ El contrato de 1 año presenta menor cantidad de clientes (166) que abandonaron la empresa a comparación con el anterior tipo de contrato.")
                    st.write("✔️ El contrato de 2 años es el que presenta la menor fuga de clientes (48).")
                
                st.write("---")

                st.subheader("Churn (var categórica) vs SeniorCitizen (var categórica)")
                st.write("🔹 Churn = Si el cliente abandonó la empresa (Yes/No)")
                st.write("🔹 SeniorCitizen = Si el cliente es adulto mayor")
                # CREAR 2 COLUMNAS PARA MOSTRAR GRÁFICO Y TEXTO
                col3,col4 = st.columns(2)
                with col3:
                    #GENERACIÓN DE GRÁFICO
                    fig, ax = plt.subplots()
                    ax.set_title("Mapa de calor: Fuga de clientes según sean o no adultos mayores", fontweight="bold")

                    # CREACIÓN DE TABLA CROSS
                    contingencia = pd.crosstab(df["SeniorCitizen"], df["Churn"])

                    # CREACIÓN DE GRÁFICO SEGÚN TABLA CROSS
                    sns.heatmap(contingencia, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                    ax.set_xlabel("Cliente abandonó la empresa")
                    ax.set_ylabel("Adulto mayor (0=No, 1=Sí)")
                    st.pyplot(fig)

                with col4:
                    #INTERPRETACIÓN
                    st.write("**INTERPRETACIÓN:**")
                    st.write("❗ Al comparar los 2 grupos de clientes (adultos mayores y no mayores), se aprecia que la fuga de clientes se dio más por parte de aquellos adultos no mayores de edad (1393 vs 476).")
                    st.write("Sin embargo, al analizar estos 2 grupos de forma independiente: ")
                    st.write("👨 **Adultos no mayores:** Más de la mitad de los clientes permanecen en la empresa.")
                    st.write("👴 **Adultos mayores:** La cantidad de clientes que permanecen en la empresa no difiere mucho a la cantidad de clientes que se fueron.")

            # DESARROLLO DE ITEM 9
            with tab9:
                st.title("Ítem 9️⃣: Análisis basado en parámetros seleccionados")
                # OBTENCIÓN DE VARIABLES SELECCIONADAS EN MULTISELECT
                selected_columns = st.multiselect(
                    "Elija 2 variables:",
                    options=df.columns.tolist(),
                    default=["Contract", "Churn"]
                )

                # CREAR 2 COLUMNAS PARA MOSTRAR GRÁFICO Y TEXTO
                col1, col2 = st.columns(2)

                with col1:
                    if len(selected_columns) == 2:
                        #GENERACION DE GRÁFICO DE BARRAS AGRUPADO
                        fig, ax = plt.subplots(figsize=(8,5))
                        sns.countplot(x=selected_columns[0], hue=selected_columns[1], data=df, palette="Set2", ax=ax)
                        ax.set_title(f"Distribución de {selected_columns[1]} según {selected_columns[0]}", fontweight="bold")
                        ax.set_xlabel(selected_columns[0])
                        ax.set_ylabel("Cantidad de clientes")

                        #AGREGAR ETIQUETAS EN EL GRÁFICO
                        for p in ax.patches:
                            altura = p.get_height()
                            ax.text(
                                p.get_x() + p.get_width()/2.,   
                                altura + 0.05,                  
                                f'{int(altura)}',               
                                ha='center', va='bottom'         
                            )

                        st.pyplot(fig)
                    else:
                        #EN CASO NO ELIJA 2 VARIABLES
                        st.warning("⚠️ Elija **solo 2** variables.")

                with col2:
                    if len(selected_columns) == 2:
                        #INTEPRETACIÓN
                        st.write("**INTERPRETACIÓN:**")
                        st.write(f"ℹ️ Este gráfico muestra cómo se distribuye la variable **{selected_columns[1]}** en función de la variable **{selected_columns[0]}**.")                                

            # DESARROLLO DE ITEM 10
            with tab10:
                st.title("Ítem 🔟: Hallazgos clave")                
                #CREAR LAS COLUMNAS
                col1, col2 = st.columns(2)    
                with col1:
                    st.subheader("Visualización principal de fuga de clientes") 
                    # CREACIÓN DE FIGURA
                    fig, ax = plt.subplots()
                    sns.histplot(df["tenure"], kde=True, ax=ax)

                    # AGREGAR TÍTULO AL GRÁFICO
                    ax.set_title(f"Distribución de la cantidad de meses de permanencia", fontweight="bold")
                    ax.set_xlabel("Cantidad de meses de permanencia")
                    ax.set_ylabel("Cantidad de clientes")
                    # CREAR DEGRADADO DE COLORES EN BARRAS
                    cmap = cm.get_cmap("viridis", len(ax.patches))

                    for i, p in enumerate(ax.patches):
                        #ASIGNAR COLOR DISTINTO A CADA BARRA
                        color = cmap(i) 
                        p.set_facecolor(color)

                        # COLOCAR ETIQUETAR A CADA BARRA
                        altura = p.get_height()
                        if altura > 0:
                            ax.text(
                                p.get_x() + p.get_width() / 2,
                                altura,
                                int(altura),
                                ha="center", va="bottom",
                                fontsize=7, fontweight="bold"
                            )
                    st.pyplot(fig)

                with col2:
                   #PRINCIPALES INSIGHTS DERIVADOS DEL EDA       
                    st.subheader("Insights principales derivados del EDA")   
                    st.write("- El máximo cargo total cobrado a un cliente es de **S/8684.80**")
                    st.write("- Casi el 60% de los clientes prefieren la **facturación electrónica**.")
                    st.write("- La mayoría de los clientes que contrata servicio de Internet es con tecnología de **fibra óptica**.")
                    st.write("- **Menos del 10% del total de clientes** contrata o había contratado servicio telefónico.")
                    st.write("- Casi la mitad de los clientes que ha contratado servicio de Internet, ha contratado también **servicio técnico**.")
                    st.write("- Se cuenta con **11 clientes nuevos**, es decir que aún no cumplen 1 mes de haber contratado servicios de la empresa. Asimismo, existe una gran cantidad de clientes (1238) cuyo tiempo de permanencia varía entre 0 a 5 meses.")
                    st.write("- Del total de clientes, **más del 70%** mantiene sus contratos con la empresa.")
                    st.write("- Más de la mitad de los clientes (55.02%) tienen un contrato de tipo **Mes a Mes**.")                      
                    st.write("- El cargo mensual promedio que pagan los clientes que tienen contratos mes a mes (66.40) es mayor que el promedio de cargo mensual de los contratos por 2 años (60.77).")
                    st.write("- Los clientes suelen abandonar la empresa luego de **1 mes de permanencia**, dando de baja los servicios contratados. La mayoría de ellos tenían un contrato Mes a Mes. **Con el pasar del tiempo, la fuga de clientes se va reduciendo.**") 
                    st.write("- Una minoría del total de clientes es adulto mayor (16.21%).")
                    st.write("- La fuga de clientes ha consistido, en su mayoría, de clientes que **no son adultos mayores**.")

        except Exception as e:
            st.error("❌ No se pudo cargar el archivo.")
            st.exception(e)

    else:
        st.warning("Por favor, cargue un archivo .CSV para continuar.")

  

elif interfaz == "Conclusiones Finales 📌":
    st.title("Conclusiones Finales 📌")
    st.write("---")
    st.write ("**1. Servicios 📞 💻:**")
    st.write("De todos los servicios que ofrece la empresa, el servicio de telefonía es el menos adquirido por los clientes, esto puede deberse a la masificación de los celulares y la comunicación directa que se realiza por medio de redes sociales como Whatsapp. Ante ello, se recomienda ofrecer dicho servicio como complemento o en paquetes con otros servicios más demandados. En cuanto al servicio de Internet, priorizar la inversión en fibra óptica frente a DSL y promocionarla, ya que posee más ventajas en cuanto a velocidad y estabilidad.")
    st.write ("**2. Retención temprana de clientes 🤝:**")
    st.write("""
    Con el fin de frenar la fuga de los clientes en los primeros meses de estancia, se recomienda implementar estrategias de retención temprana como la aplicación descuentos al cumplir cierto tiempo de permanencia. Además, también se sugiere evaluar otros factores que puedan estar influyendo negativamente, tales como:  
    - Trato al cliente posterior a la venta: nivel de satisfacción de cliente luego de la atención del personal de SAC, tiempo de respuesta a consultas, quejas y/o reclamos.  
    - Calidad del servicio contratado. Por ejemplo, estabilidad.  
    - Poca diferenciación con la competencia a nivel de valor agregado: promociones, paquetes de servicios.
    """)

    st.write ("**3. Servicios adicionales como palanca de retención 🛠️:**")
    st.write ("Según el análisis realizado, se observó que casi la mitad de los clientes que contrataron el servicio de Internet, también contrataron el servicio técnico. Ante ello, se sugiere ofrecer el servicio de Internet en paquete con el servicio técnico, de tal forma que aumente la percepción de valor del cliente hacia la empresa.")
    st.write ("**4. Facturación electrónica 🧑‍💻:**")
    st.write ("Dado que casi el 60% de los clientes optan por la facturación electrónica, se debería potenciar la digitalización del proceso de facturación y difundir sus ventajas a los clientes. Esto permitirá a la empresa disminuir sus costos operativos.")
    st.write ("**5. Estructura de cargos de pago 💳:**")
    st.write ("En promedio, el cargo mensual de los contratos Mes a Mes (S/66.40) es mayor que el de los contratos de 2 años (S/60.77), lo que evidencia que los clientes prefieren contratos flexibles ('sin ataduras') en comparación con el ahorro. Ante ello, se recomienda diseñar contratos intermedios (de 6 meses, por ejemplo) con beneficios frente al contrato Mes a Mes, en donde se resalte el ahorro que trae consigo.")