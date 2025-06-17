# ai_classifier.py - Clasificador de aeronaves con IA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2


class ClasificadorAeronaves:
    def __init__(self):
        self.model = None
        self.class_names = ['Boeing-737', 'Airbus-A320', 'Cessna-172', 'Embraer-190', 'ATR-72']
        self.img_height = 224
        self.img_width = 224
        
    def crear_modelo(self):
        """Crear modelo CNN simple para clasificación"""
        self.model = keras.Sequential([
            layers.Rescaling(1./255),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Modelo creado exitosamente")
        return self.model
    
    def entrenar_modelo(self, ruta_datos):
        """Entrenar el modelo con imágenes organizadas en carpetas"""
        if not os.path.exists(ruta_datos):
            print(f"❌ Error: La ruta {ruta_datos} no existe")
            return False
            
        try:
            # Crear datasets de entrenamiento y validación
            train_ds = tf.keras.utils.image_dataset_from_directory(
                ruta_datos,
                validation_split=0.2,
                subset="training",
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=32
            )
            
            val_ds = tf.keras.utils.image_dataset_from_directory(
                ruta_datos,
                validation_split=0.2,
                subset="validation", 
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=32
            )
            
            # Obtener nombres de clases del dataset
            self.class_names = train_ds.class_names
            print(f"📂 Clases encontradas: {self.class_names}")
            
            # Optimizar rendimiento
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
            
            # Crear modelo si no existe
            if self.model is None:
                self.crear_modelo()
            
            # Entrenar modelo
            print("🚀 Iniciando entrenamiento...")
            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=10,  # Pocas épocas para prueba rápida
                verbose=1
            )
            
            # Guardar modelo
            self.guardar_modelo()
            print("✅ Entrenamiento completado y modelo guardado")
            return True
            
        except Exception as e:
            print(f"❌ Error durante entrenamiento: {str(e)}")
            return False
    
    def predecir_imagen(self, ruta_imagen):
        """Predecir tipo de aeronave desde imagen"""
        if self.model is None:
            if not self.cargar_modelo():
                return None, 0
        
        try:
            # Cargar y preprocesar imagen
            img = tf.keras.utils.load_img(ruta_imagen, target_size=(self.img_height, self.img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            # Hacer predicción
            predictions = self.model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])
            
            # Obtener resultado
            clase_predicha = self.class_names[np.argmax(score)]
            confianza = 100 * np.max(score)
            
            return clase_predicha, confianza
            
        except Exception as e:
            print(f"❌ Error en predicción: {str(e)}")
            return None, 0
    
    def guardar_modelo(self):
        """Guardar modelo entrenado"""
        if self.model:
            self.model.save('modelo_aeronaves.h5')
            # Guardar nombres de clases
            with open('clases_aeronaves.txt', 'w') as f:
                for clase in self.class_names:
                    f.write(f"{clase}\n")
    
    def cargar_modelo(self):
        """Cargar modelo previamente entrenado"""
        try:
            if os.path.exists('modelo_aeronaves.h5'):
                self.model = tf.keras.models.load_model('modelo_aeronaves.h5')
                
                # Cargar nombres de clases
                if os.path.exists('clases_aeronaves.txt'):
                    with open('clases_aeronaves.txt', 'r') as f:
                        self.class_names = [line.strip() for line in f.readlines()]
                
                print("✅ Modelo cargado exitosamente")
                return True
            else:
                print("❌ No se encontró modelo entrenado")
                return False
        except Exception as e:
            print(f"❌ Error cargando modelo: {str(e)}")
            return False

class VentanaIAAeronaves(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("IA - Clasificador de Aeronaves")
        self.geometry("700x650")
        self.configure(bg='#ecf0f1')
        
        self.clasificador = ClasificadorAeronaves()
        self.ruta_imagen_seleccionada = None
        
        self.crear_interfaz()
        
        # Intentar cargar modelo existente
        if self.clasificador.cargar_modelo():
            self.label_estado.config(text="✅ Modelo cargado - Listo para clasificar", fg='green')
        else:
            self.label_estado.config(text="⚠️ No hay modelo entrenado - Entrena primero", fg='orange')
    
    def crear_interfaz(self):
        # Título
        titulo = tk.Label(self, text="🤖 Clasificador IA de Aeronaves", 
                         font=('Arial', 14, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        titulo.pack(pady=10)
        
        # Estado del modelo
        self.label_estado = tk.Label(self, text="Verificando modelo...", 
                                   font=('Arial', 10), bg='#ecf0f1')
        self.label_estado.pack(pady=10)
        
        # Frame para entrenamiento
        frame_entrenamiento = tk.LabelFrame(self, text="🚀 Entrenamiento", 
                                          font=('Arial', 12, 'bold'), bg='#ecf0f1')
        frame_entrenamiento.pack(fill='x', padx=10, pady=10)
        
        instrucciones = tk.Label(frame_entrenamiento, 
                               text="📁imágenes en carpetas Organizadas por tipo:\n" +
                                    "📂 aeronaves/\n" +
                                    "   ├── Boeing-737/\n" +
                                    "   ├── Airbus-A320/\n" +
                                    "   ├── Cessna-172/\n" +
                                    "   └── ...etc, ",
                               font=('Arial', 5), bg='#ecf0f1', justify='left')
        instrucciones.pack(pady=10)
        
        btn_entrenar = tk.Button(frame_entrenamiento, text="📂 Seleccionar Carpeta y Entrenar",
                               command=self.entrenar_modelo, bg='#3498db', fg='white',
                               font=('Arial', 10), height=2)
        btn_entrenar.pack(pady=10)
        
        # Frame para clasificación
        frame_clasificacion = tk.LabelFrame(self, text="🔍 Clasificación", 
                                          font=('Arial', 12, 'bold'), bg='#ecf0f1')
        frame_clasificacion.pack(fill='x', padx=10, pady=10)
        
        btn_seleccionar = tk.Button(frame_clasificacion, text="🖼️ Seleccionar Imagen",
                                  command=self.seleccionar_imagen, bg='#2ecc71', fg='white',
                                  font=('Arial', 12), height=2)
        btn_seleccionar.pack(pady=10)
        
        self.label_imagen = tk.Label(frame_clasificacion, text="No hay imagen seleccionada",
                                   bg='#ecf0f1', font=('Arial', 11))
        self.label_imagen.pack(pady=5)
        
        btn_clasificar = tk.Button(frame_clasificacion, text="🎯 Clasificar Aeronave",
                                 command=self.clasificar_imagen, bg='#e74c3c', fg='white',
                                 font=('Arial', 12), height=2)
        btn_clasificar.pack(pady=10)
        
        # Frame para resultados
        frame_resultados = tk.LabelFrame(self, text="📊 Resultados", 
                                       font=('Arial', 10, 'bold'), bg='#ecf0f1')
        frame_resultados.pack(fill='x', padx=10, pady=10)
        
        self.label_resultado = tk.Label(frame_resultados, text="Esperando clasificación...",
                                      font=('Arial', 12), bg='#ecf0f1')
        self.label_resultado.pack(pady=10)
        
        # Botón para usar resultado en registro
        self.btn_usar_resultado = tk.Button(frame_resultados, 
                                          text="📋 Usar en Registro de Aeronave",
                                          command=self.usar_en_registro, 
                                          bg='#9b59b6', fg='white',
                                          font=('Arial', 12), state='disabled')
        self.btn_usar_resultado.pack(pady=10)
        
        self.ultimo_resultado = None
    
    def entrenar_modelo(self):
        """Entrenar modelo con imágenes"""
        carpeta = filedialog.askdirectory(title="Seleccionar carpeta con imágenes de aeronaves")
        if not carpeta:
            return
        
        self.label_estado.config(text="🔄 Entrenando modelo... (puede tomar varios minutos)", fg='blue')
        self.update()
        
        # Entrenar en thread separado para no bloquear UI
        import threading
        
        def entrenar():
            exito = self.clasificador.entrenar_modelo(carpeta)
            if exito:
                self.label_estado.config(text="✅ Modelo entrenado exitosamente", fg='green')
                messagebox.showinfo("Éxito", "¡Modelo entrenado correctamente!\nYa puedes clasificar aeronaves.")
            else:
                self.label_estado.config(text="❌ Error en entrenamiento", fg='red')
                messagebox.showerror("Error", "No se pudo entrenar el modelo.\nVerifica que las imágenes estén organizadas correctamente.")
        
        thread = threading.Thread(target=entrenar)
        thread.daemon = True
        thread.start()
    
    def seleccionar_imagen(self):
        """Seleccionar imagen para clasificar"""
        tipos = [('Imágenes', '*.png *.jpg *.jpeg *.gif *.bmp')]
        archivo = filedialog.askopenfilename(title="Seleccionar imagen de aeronave", filetypes=tipos)
        
        if archivo:
            self.ruta_imagen_seleccionada = archivo
            nombre_archivo = os.path.basename(archivo)
            self.label_imagen.config(text=f"📷 Imagen: {nombre_archivo}")
    
    def clasificar_imagen(self):
        """Clasificar imagen seleccionada"""
        if not self.ruta_imagen_seleccionada:
            messagebox.showwarning("Advertencia", "Primero selecciona una imagen")
            return
        
        if self.clasificador.model is None:
            messagebox.showerror("Error", "No hay modelo entrenado disponible")
            return
        
        self.label_resultado.config(text="🔄 Clasificando...")
        self.update()
        
        # Clasificar imagen
        tipo_predicho, confianza = self.clasificador.predecir_imagen(self.ruta_imagen_seleccionada)
        
        if tipo_predicho:
            resultado_texto = f"🎯 Tipo detectado: {tipo_predicho}\n📊 Confianza: {confianza:.1f}%"
            color = 'green' if confianza > 70 else 'orange' if confianza > 50 else 'red'
            
            self.label_resultado.config(text=resultado_texto, fg=color)
            self.ultimo_resultado = tipo_predicho
            self.btn_usar_resultado.config(state='normal')
            
            # Mostrar información adicional
            info_adicional = self.obtener_info_aeronave(tipo_predicho)
            if info_adicional:
                messagebox.showinfo("Información Detectada", info_adicional)
        else:
            self.label_resultado.config(text="❌ Error en clasificación", fg='red')
            self.btn_usar_resultado.config(state='disabled')
    
    def obtener_info_aeronave(self, tipo):
        """Obtener información adicional según el tipo detectado"""
        info_aeronaves = {
            'Boeing-737': {
                'fabricante': 'Boeing',
                'categoria': 'Pesada',
                'peso_aprox': 79000
            },
            'Airbus-A320': {
                'fabricante': 'Airbus', 
                'categoria': 'Pesada',
                'peso_aprox': 73500
            },
            'Cessna-172': {
                'fabricante': 'Cessna',
                'categoria': 'Liviana', 
                'peso_aprox': 1157
            },
            'Embraer-190': {
                'fabricante': 'Embraer',
                'categoria': 'Mediana',
                'peso_aprox': 51800
            },
            'ATR-72': {
                'fabricante': 'ATR',
                'categoria': 'Mediana',
                'peso_aprox': 22500
            }
        }
        
        if tipo in info_aeronaves:
            info = info_aeronaves[tipo]
            return f"Fabricante: {info['fabricante']}\nCategoría: {info['categoria']}\nPeso MTOW aproximado: {info['peso_aprox']:,} kg"
        return None
    
    def usar_en_registro(self):
        """Usar resultado en el formulario de registro"""
        if not self.ultimo_resultado:
            return
        
        # Cerrar ventana actual y abrir registro con datos pre-llenados
        self.destroy()
        
        # Abrir ventana de registro con datos sugeridos
        ventana_registro = VentanaRegistroAeronaveIA(self.parent, self.ultimo_resultado)

class VentanaRegistroAeronaveIA(tk.Toplevel):
    """Ventana de registro con datos sugeridos por IA"""
    def __init__(self, parent, tipo_detectado):
        super().__init__(parent)
        self.parent = parent
        self.tipo_detectado = tipo_detectado
        self.title("Registrar Aeronave - Con IA")
        self.geometry("600x450")
        self.configure(bg='#ecf0f1')
        
        # Variables
        self.var_matricula = tk.StringVar()
        self.var_modelo = tk.StringVar()
        self.var_fabricante = tk.StringVar()
        self.var_peso_mtow = tk.StringVar()
        self.var_horas_vuelo = tk.StringVar()
        self.var_hangar = tk.StringVar()
        
        self.crear_interfaz()
        self.prellenar_datos()
    
    def crear_interfaz(self):
        # Título con IA
        titulo_frame = tk.Frame(self, bg='#ecf0f1')
        titulo_frame.pack(pady=20)
        
        tk.Label(titulo_frame, text="🤖 Registro Asistido por IA", 
                font=('Arial', 14, 'bold'), bg='#ecf0f1', fg='#2c3e50').pack()
        tk.Label(titulo_frame, text=f"Tipo detectado: {self.tipo_detectado}", 
                font=('Arial', 12), bg='#ecf0f1', fg='#27ae60').pack()
        
        # Frame principal
        main_frame = tk.Frame(self, bg='#ecf0f1')
        main_frame.pack(padx=30, pady=16, fill='both', expand=True)
        
        # Campos del formulario
        campos = [
            ("Matrícula (CP-XXXX):", self.var_matricula),
            ("Modelo:", self.var_modelo),
            ("Fabricante:", self.var_fabricante),
            ("Peso MTOW (kg):", self.var_peso_mtow),
            ("Horas de Vuelo:", self.var_horas_vuelo)
        ]
        
        for i, (label_text, var) in enumerate(campos):
            tk.Label(main_frame, text=label_text, font=('Arial', 12), 
                    bg='#ecf0f1', fg='#2c3e50').grid(row=i, column=0, sticky='w', pady=8)
            
            entry = tk.Entry(main_frame, textvariable=var, font=('Arial', 12), width=25)
            entry.grid(row=i, column=1, padx=20, pady=8)
            
            # Resaltar campos pre-llenados por IA
            if i in [1, 2, 3]:  # modelo, fabricante, peso
                entry.configure(bg='#d5f4e6')  # Verde claro
        
        # Selección de hangar
        tk.Label(main_frame, text="Hangar:", font=('Arial', 12), 
                bg='#ecf0f1', fg='#2c3e50').grid(row=len(campos), column=0, sticky='w', pady=8)
        
        hangares = self.parent.db.obtener_hangares()
        hangar_values = [f"{h[1]} - {h[2]}" for h in hangares]
        
        hangar_combo = ttk.Combobox(main_frame, textvariable=self.var_hangar, 
                                   values=hangar_values, font=('Arial', 12), width=23)
        hangar_combo.grid(row=len(campos), column=1, padx=20, pady=8)
        
        # Nota sobre IA
        nota_frame = tk.Frame(main_frame, bg='#e8f6f3', relief='solid', bd=1)
        nota_frame.grid(row=len(campos)+1, column=0, columnspan=2, pady=20, padx=10, sticky='ew')
        
        tk.Label(nota_frame, text="🤖 Los campos en verde fueron sugeridos por IA", 
                font=('Arial', 10, 'italic'), bg='#e8f6f3', fg='#27ae60').pack(pady=5)
        
        # Botones
        btn_frame = tk.Frame(main_frame, bg='#ecf0f1')
        btn_frame.grid(row=len(campos)+2, column=0, columnspan=2, pady=20)
        
        tk.Button(btn_frame, text="💾 Guardar", command=self.guardar_aeronave,
                 bg='#2ecc71', fg='white', font=('Arial', 12), width=12).pack(side='left', padx=10)
        tk.Button(btn_frame, text="❌ Cancelar", command=self.destroy,
                 bg='#e74c3c', fg='white', font=('Arial', 12), width=12).pack(side='right', padx=10)
    
    def prellenar_datos(self):
        """Pre-llenar datos basados en el tipo detectado por IA"""
        datos_sugeridos = {
            'Boeing-737': {
                'modelo': 'Boeing 737-800',
                'fabricante': 'Boeing',
                'peso_mtow': '79000'
            },
            'Airbus-A320': {
                'modelo': 'Airbus A320',
                'fabricante': 'Airbus', 
                'peso_mtow': '73500'
            },
            'Cessna-172': {
                'modelo': 'Cessna 172',
                'fabricante': 'Cessna',
                'peso_mtow': '1157'
            },
            'Embraer-190': {
                'modelo': 'Embraer 190',
                'fabricante': 'Embraer',
                'peso_mtow': '51800'
            },
            'ATR-72': {
                'modelo': 'ATR 72',
                'fabricante': 'ATR',
                'peso_mtow': '22500'
            }
        }
        
        if self.tipo_detectado in datos_sugeridos:
            datos = datos_sugeridos[self.tipo_detectado]
            self.var_modelo.set(datos['modelo'])
            self.var_fabricante.set(datos['fabricante'])
            self.var_peso_mtow.set(datos['peso_mtow'])
    
    def guardar_aeronave(self):
        """Guardar aeronave con datos asistidos por IA"""
        # Reutilizar lógica de validación de la ventana original
        if not all([self.var_matricula.get(), self.var_modelo.get(), 
                   self.var_fabricante.get(), self.var_peso_mtow.get(),
                   self.var_horas_vuelo.get(), self.var_hangar.get()]):
            messagebox.showerror("Error", "Todos los campos son obligatorios")
            return
        
        try:
            peso_mtow = float(self.var_peso_mtow.get())
            horas_vuelo = float(self.var_horas_vuelo.get())
        except ValueError:
            messagebox.showerror("Error", "Peso y horas de vuelo deben ser números válidos")
            return
        
        # Obtener ID del hangar
        hangar_seleccionado = self.var_hangar.get().split(" - ")[0]
        hangar = self.parent.db.obtener_hangar_por_nombre(hangar_seleccionado)
        
        if not hangar:
            messagebox.showerror("Error", "Hangar no válido")
            return
        
        # Insertar en la base de datos
        success = self.parent.db.insertar_aeronave(
            matricula=self.var_matricula.get(),
            modelo=self.var_modelo.get(),
            fabricante=self.var_fabricante.get(),
            peso_mtow=peso_mtow,
            categoria=self.parent.categorizar_aeronave(peso_mtow),
            horas_vuelo=horas_vuelo,
            hangar_id=hangar[0]
        )
        
        if success:
            messagebox.showinfo("Éxito", "✅ Aeronave registrada con asistencia de IA")
            self.destroy()
        else:
            messagebox.showerror("Error", "Matrícula ya existe en el sistema")