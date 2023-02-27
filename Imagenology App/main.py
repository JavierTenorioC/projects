from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
#from kivy.uix.widget import Widget
from kivy.lang import Builder 
from kivy.config import Config
from kivy.utils import platform
from kivy.uix.slider import Slider
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image



Config.set("graphics","width",800)
Config.set("graphics","height",1000)



#designar la ruta del archivo kv
#Builder.load_file('ruta y nombre del archivo')


KV = '''
#: import utils kivy.utils
#: import Factory kivy.factory.Factory
<MyPopup@Popup>:
    auto_dismiss: True
    title: "Ayuda sobre el procesamiento"
    size_hint: 1,1
    
    BoxLayout: 
        orientation: "vertical"
        Label:
            text: '---------Descripcion de los Procesamientos---------\\n Grises: Conversion a escala de grises: convierte tu imagen \\n a escala de grises. \\n Otsu: Multiumbralizacion Otsu: umbraliza de (2-4 tonos) tu imagen.\\n E. RGB: Ecualizacion RGB: ecualiza los canales rgb de la imagen. \\n Inv.G.: Invertir colores W/B: invierte los colores en la escala de grises.\\n E.I.: Ecualización de iluminacion: ecualiza unicamente la componente\\n de iluminacion. \\n H.B.: HighBoost RGB: Realiza un filtro HB para mejorar los detalles \\n de la imagen\\n introduciendo un factor de amplificacion A>=1 \\n con una mascara de 3er orden (A=2 recomendado). \\n E.G.:Ecualizacion Gamma: (0-2 recomendado) \\n Fuzzy: aplica equalización difusa a imágenes en escala de grises'
        Label:
            text: '---------Descripcion de los botones de direccion---------\\n CLC:Limpia el visor de imagen. \\n Carpeta: Ve a carpeta de imágenes a procesar. \\n Img: Ve las imagenes que procesaste.'
                                    
                
        Button:
            text: "Cerrar"
            on_press: root.dismiss()
            background_color: 1,0,0    

          
<contenedor>:
    id: my_widget
    spacing:30


    canvas:
        Color:
            rgb: 0.03,0.03,0.18
        Rectangle:
            size: self.size
            pos: self.pos
    

    GridLayout:
        cols : 1 

        canvas:
            Color:
                rgb: 0,0,1
            Rectangle:
                size: self.size
                pos: self.pos
                
                
        GridLayout:
            cols : 1
            canvas:
                Color:
                    rgba:  0.03,0.03,0.1,1
                Rectangle:
                    size: self.size
                    pos: self.pos
                    
            Image: 
                id: my_image
                source: ""
        
            FileChooserIconView:
                path: './' #directorio donde se buscan las imagenes
                id: filechooser
                on_selection: my_widget.selected(filechooser.selection)
        GridLayout:
            cols : 3
            size_hint: 1, None
            height: 35 * 5
            
            GridLayout:
                cols : 3
                
                Button: #1
                    size_hint: 1,1
                    text: "Grises"
                    on_press: my_widget.btnProcesar(filechooser.selection)
                    background_color:
                        utils.get_color_from_hex('#FA0000')
                Button: #3
                    text : "E. RGB"
                    on_press: my_widget.eqRGB(filechooser.selection)
                    background_color:
                        utils.get_color_from_hex('#94FA00')  
                Button: #4
                    text : "Inv. G."
                    on_press: my_widget.invertirColores(filechooser.selection)
                    background_color:
                        utils.get_color_from_hex('#800080')
                Button: #2
                    text : "Otsu"
                    on_press: my_widget.btnOtsu(filechooser.selection,sOtsu.value)
                
                Slider:
                    id: sOtsu
                    min: 2
                    max: 4
                    step: 1
                    orientation: 'horizontal'
                    cursor_height : 30
                    cursor_width : 30
                Label:
                    text : str(sOtsu.value)
                
                Button:
                    text : 'HB'
                    on_press : my_widget.highBoost(filechooser.selection,hb.text)
                    background_color:
                        utils.get_color_from_hex('#FAFA00')
                Label:
                    text : 'A>=1'
                    background_color:
                        utils.get_color_from_hex('#FAFA00')
                TextInput:
                    id:hb
                    multiline: False
                    text: "2"
                    halign : 'center'
                    valign : 'center'
                    background_color:
                        utils.get_color_from_hex('#FAFA00')
                Button: # 5
                    on_press:  my_widget.eqIlum(filechooser.selection)  
                    text: "E.I."
                    background_color:
                        utils.get_color_from_hex('#FA9000')
                Button:
                    text : "Carpeta"
                    on_press:  my_widget.direccionar() 
                    background_color:
                        utils.get_color_from_hex('#0085FA')
                Button:
                    text : "Img"
                    on_press:  my_widget.imagenes()  
                    background_color:
                        utils.get_color_from_hex('#0085FA')
            
            GridLayout:
                cols : 3
                Button:
                    text : 'Fuzzy'
                    on_press: my_widget.fuzz(filechooser.selection,s1.value,s2.value,s3.value)
                Button: 
                    text : 'clc'
                    on_press: my_widget.limpiar()
                    background_color:
                        utils.get_color_from_hex('#0085FA')   
                Button:
                    text : '?'
                    on_press:  Factory.MyPopup().open()
                    background_color:
                        utils.get_color_from_hex('#FA0075')
                Label:
                    text : 's1'
                Slider:
                    id: s1
                    min: 0
                    max: 255
                    step: 1
                    orientation: 'horizontal'
                    value : 30
                    cursor_height : 30
                    cursor_width : 30
                Label:
                    text : str(int(s1.value)) 
                Label:
                    text : 's2'
                Slider:
                    id: s2
                    min: 0
                    max: 255
                    step: 1
                    orientation: 'horizontal'
                    value : 220
                    cursor_height : 30
                    cursor_width : 30
                Label:
                    text : str(int(s2.value)) 
                Label:
                    text : 's3'
                Slider:
                    id: s3
                    min: 0
                    max: 255
                    step: 1
                    orientation: 'horizontal'
                    value: 245
                    cursor_height : 30
                    cursor_width : 30
                Label:
                    text : str(int(s3.value)) 
                
'''

Builder.load_string(KV) #cargar el todo el tecto KV             
                    


class contenedor(BoxLayout):#herencia de las funciones de boxlayout
    contador = 0

    def selected(self,filename):
        try:            
            
            self.ids.my_image.source = filename[0] #aqui file name tiene la direccion
        except:
            print("error")
        
    def btnProcesar(self,filename):        
        try:
            self.contador = self.contador + 1
            imagen = Image.open(filename[0])
            imagen = np.array(imagen)
            gris = (0.299*imagen[:,:,0]) + (0.587*imagen[:,:,1]) + (0.114*imagen[:,:,2])  # asi persive el humano
            gris = Image.fromarray((gris).astype(np.uint8))        
            gris.save('./Grises' + str(self.contador) + '.jpg')
            self.ids.my_image.source = './Grises' + str(self.contador) + '.jpg'
        except:
            print("error")
            
    def btnOtsu(self,filename,clases):
        try:
            self.contador = self.contador + 1
            imagen = Image.open(filename[0])
            imagen = np.array(imagen)
            gris = (0.299*imagen[:,:,0]) + (0.587*imagen[:,:,1]) + (0.114*imagen[:,:,2])  # asi persive el humano
            histo = np.zeros(256)
            filas = gris.shape[0]
            columnas = gris.shape[1]
            
            
            for i in range (filas):
                for j in range (columnas):
                    pixel = int(gris[i,j])
                    histo[pixel]+=1
                    

            
            Pro = histo/(filas*columnas)
            #Calculo de la ec 22
            P = np.zeros(256)
            S = np.zeros(256)
            P[0] = Pro[0]
            S[0] = 0*Pro[0]
            
            for v in range(len(Pro)-1):
                P[v+1] = P[v]+Pro[v+1] #Ec 22
                S[v+1] = S[v]+(v+1)*(Pro[v+1])#Ec 23
                
            PP = np.zeros((256, 256))    
            SS = np.zeros((256, 256))
            HH = np.zeros((256, 256))
            resta1 = np.zeros(len(Pro)+2)
            resta2 = np.zeros(len(Pro)+2)
            resta1[1:-1] = P
            resta2[1:-1] = S
            for u in range (256):
                for v in range(256):
                    PP[u,v] = P[v]-resta1[u]+0.0000000001#Ec 24
                    SS[u,v] = S[v]-resta2[u]#Ec 25       
                    HH[u,v] = (SS[u,v]**2)/(PP[u,v]) #Ec 29
                    
            u = 0
            cla = np.round(int(clases))
            L = 255
            
            imgTrans = np.zeros((filas,columnas))
            if(cla == 2): #dos clases
                for t1 in range (0,L-(cla-1),1):
                    r1 = HH[0,t1]+HH[t1+1,L]
                    if (u < r1):
                        u = r1
                        umbral = t1-1 
            
            
                
                for i in range(filas):
                    for j in range(columnas):
                        if(gris[i,j] < umbral):
                            imgTrans[i,j] = umbral
                        else:
                            imgTrans[i,j] = 255
                        
            elif(cla == 3):#3clases
                for t1 in range (0,L-(cla-1),1):
                    for t2 in range (t1+1,L-(cla-2),1):
                        r1 = HH[1,t1]+HH[t1+1,t2]+HH[t2+1,L]#suma de todas las clases
                        if (u < r1):
                            u = r1
                            umbral = np.array([t1,t2])-1
                            
                #umbralizacion
                for i in range(filas):
                    for j in range(columnas):
                        if(gris[i,j] < umbral[0]):
                            imgTrans[i,j] = umbral[0]
                        elif(gris[i,j]>=umbral[0] and gris[i,j]<umbral[1]):
                            imgTrans[i,j] = umbral[1]
                        else:
                            imgTrans[i,j] = 255
                        
                   
            elif(cla == 4):
                for t1 in range (0,L-(cla-1),1):
                    for t2 in range (t1+1,L-(cla-2),1):
                        for t3 in range (t2+1,L-(cla-3),1):
                            r1 = HH[2,t1]+HH[t2+1,t3]+HH[t1+1,t2]+HH[t3+1,L]#suma de todas las clases
                            if (u < r1):
                                u = r1
                                umbral = np.array([t1,t2,t3])-1
                                
                    #umbralizacion
                for i in range(filas):
                    for j in range(columnas):
                        if(gris[i,j] < umbral[0]):
                            imgTrans[i,j] = umbral[0]
                        elif(gris[i,j]>=umbral[0] and gris[i,j]<umbral[1]):
                            imgTrans[i,j] = umbral[1]
                        elif(gris[i,j]>=umbral[1] and gris[i,j]<umbral[2]):
                            imgTrans[i,j] = umbral[2]
                        else:
                            imgTrans[i,j] = 255
                else:
                    print("error")
                
            
            imgTrans = np.uint8(imgTrans)
            imgTrans = Image.fromarray((imgTrans*255).astype(np.uint8))        
            imgTrans.save('./MOtsu' + str(self.contador) + '.jpg')
            
            self.ids.my_image.source = "./MOtsu"+ str(self.contador)+ '.jpg'
        except:
            print("error")
        
        
        
    def limpiar(self):
        try:
            self.ids.filechooser._update_files()
            self.ids.my_image.source = ""  
        except:
            print("error")
        
          
          
    def direccionar(self):
        try:
            self.ids.filechooser.path = "./"
        except:
            print("error")
    
    def imagenes(self):
        try:
            self.ids.filechooser.path = "./"
        except:
            print("error")
    def eqRGB(self,filename):
        try:
            self.contador = self.contador + 1
            imagen = Image.open(filename[0])
            imagen = np.array(imagen)
            
            
            retina = imagen
            retinaRGB = np.copy(retina)
            
            canales = [retinaRGB[:,:,0],retinaRGB[:,:,1],retinaRGB[:,:,2]]
            
            filas = retinaRGB.shape[0]
            columnas= retinaRGB.shape[1]
            salR = np.zeros((filas,columnas))
            salG = np.zeros((filas,columnas))
            salB = np.zeros((filas,columnas))
            salida = [salR,salG,salB]
            
            
            #variables a utilizar
            histoR = np.zeros(256)
            histoG = np.zeros(256)
            histoB = np.zeros(256)
            
            proR = np.zeros(256)
            proG = np.zeros(256)
            proB = np.zeros(256)
            
            equalizaR = np.zeros(256)
            equalizaG = np.zeros(256)
            equalizaB = np.zeros(256)
            
            equaliza = [equalizaR,equalizaG,equalizaB]
            histo = [histoR,histoG,histoB]
            pro = [proR,proG,proB]
            
            
            
            for numCanal in range(3):
            
                
                
                
                for i in range(filas):
                    for j in range(columnas):
                        valorPixel = canales[numCanal][i,j]
                        histo[numCanal][valorPixel] = histo[numCanal][valorPixel] + 1
                    
                pro[numCanal] = histo[numCanal]/(filas*columnas)
                
                
                acum = 0
                
                for k in range(256):
                    acum = acum + pro[numCanal][k]
                    equaliza[numCanal][k] = acum*255.0
                    
                
                for i in range(filas):
                    for j in range(columnas):
                        entrada = canales[numCanal][i,j]
                        salida[numCanal][i,j] = equaliza[numCanal][entrada]
                        
                      
            #reconstruyendo canales    
            imgEq = np.zeros((filas,columnas,3))    
            imgEq[:,:,0] = salida[0]
            imgEq[:,:,1] = salida[1]
            imgEq[:,:,2] = salida[2]
                
            imgEq = np.uint8(imgEq)
            imgEq = Image.fromarray((imgEq).astype(np.uint8))
            imgEq.save('./Ecualizada' + str(self.contador)+'.jpg')       
            self.ids.my_image.source = './Ecualizada' + str(self.contador)+ '.jpg'
        except:
            print("error")
        

                  
    def gamma(self,filename):
        try:
            self.contador = self.contador + 1
            imagen = Image.open(filename[0])
            imagen = np.array(imagen)            
            gris = (0.299*imagen[:,:,0]) + (0.587*imagen[:,:,1]) + (0.114*imagen[:,:,2])  # asi persive el humano
            gamma=1.7
            sal=gris**(1/gamma)#controla nivel de oscuridad
            imgEq = np.uint8(sal)
            imgEq = Image.fromarray((imgEq).astype(np.uint8))
            imgEq.save('./GammaEq'+ str(self.contador)+ '.jpg')       
            self.ids.my_image.source = '.GammaEq'+ str(self.contador)+ '.jpg'
        except:
            print("error")
        
        
        
        
    def invertirColores(self,filename):
        try:
            self.contador = self.contador + 1
            imagen = Image.open(filename[0])
            imagen = np.array(imagen)            
            gris = (0.299*imagen[:,:,0]) + (0.587*imagen[:,:,1]) + (0.114*imagen[:,:,2])  # asi persive el humano
            inv=255-gris
            imgEq = np.uint8(inv)
            imgEq = Image.fromarray((imgEq).astype(np.uint8))
            imgEq.save(r'./Invertidos'+ str(self.contador)+ '.jpg')       
            self.ids.my_image.source = r'./Invertidos'+ str(self.contador)+ '.jpg'
        except:
            print("error")
        
    def eqIlum(self,filename):
        try:
            self.contador = self.contador + 1
            rgb = Image.open(filename[0])
            rgb = np.array(rgb)
            
            #proceso de conversion 
            input_shape = rgb.shape
            rgb = rgb.reshape(-1, 3)
            r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            
            maxc = np.maximum(np.maximum(r, g), b)
            minc = np.minimum(np.minimum(r, g), b)
            v = maxc
            
            deltac = maxc - minc
            s = deltac / maxc
            deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
            rc = (maxc - r) / deltac
            gc = (maxc - g) / deltac
            bc = (maxc - b) / deltac
            
            h = 4.0 + gc - rc
            h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
            h[r == maxc] = bc[r == maxc] - gc[r == maxc]
            h[minc == maxc] = 0.0
            
            h = (h / 6.0) % 1.0
            res = np.dstack([h, s, v])
            
            
            hsv = res.reshape(input_shape)
            HSV = hsv
            
            #ecualizacion del canal del V
            his = np.zeros(256)
            pro = np.zeros(256)
            for i in range (HSV.shape[0]):
                for j in range (HSV.shape[1]):
                    pos = int(HSV[i,j,2])
                    his [pos] = his [pos] + 1
            
            
            pro = his/(HSV.shape[0]*HSV.shape[1])
            ecu = pro.cumsum()
            
            sal = np.zeros((HSV.shape[0], HSV.shape[1]))
            for i in range (HSV.shape[0]):
                for j in range (HSV.shape[1]):
                    pos = int(HSV[i,j,2])
                    sal[i,j] = np.uint8(ecu[pos]*255)
            
            mejoradagris = np.copy(HSV)
            mejoradagris[:,:,2] = sal
            
            
            
            #transformacion a rgb 
            hsv = mejoradagris/255.0
            
            input_shape = hsv.shape
            hsv = hsv.reshape(-1, 3)
            h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
            
            i = np.int32(h * 6.0)
            f = (h * 6.0) - i
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))
            i = i % 6
            
            rgb = np.zeros_like(hsv)
            v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)
            rgb[i == 0] = np.hstack([v, t, p])[i == 0]
            rgb[i == 1] = np.hstack([q, v, p])[i == 1]
            rgb[i == 2] = np.hstack([p, v, t])[i == 2]
            rgb[i == 3] = np.hstack([p, q, v])[i == 3]
            rgb[i == 4] = np.hstack([t, p, v])[i == 4]
            rgb[i == 5] = np.hstack([v, p, q])[i == 5]
            rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]
            
            rgbNew = rgb.reshape(input_shape)
            imgEq = np.uint8(rgbNew*255)
            imgEq = Image.fromarray((imgEq).astype(np.uint8))        
            imgEq.save(r'./Ilum'+ str(self.contador)+ '.jpg')       
            self.ids.my_image.source = r'./Ilum'+ str(self.contador)+ '.jpg'
        except:
            print("error")
  
    def gamma(self,filename,val):
        try:
            gammaVal = float(val)
            print(val)
            print(filename)
            self.contador = self.contador + 1
            imagen = Image.open(filename[0])
            imagen = np.array(imagen)            
            gris = (0.299*imagen[:,:,0]) + (0.587*imagen[:,:,1]) + (0.114*imagen[:,:,2])  # asi persive el humano        
            sal=gris**(1/gammaVal)#controla nivel de oscuridad
            imgEq = np.uint8(sal)
            imgEq = Image.fromarray((imgEq).astype(np.uint8))
            imgEq.save('./GammaEq'+ str(self.contador)+ '.jpg')       
            self.ids.my_image.source ='./GammaEq'+ str(self.contador)+ '.jpg'
        except:
            print("error")
        
    def highBoost(self,filename,val):
        try:
            self.contador = self.contador + 1
           #Adquisicion de la imagen
            rgb = Image.open(filename[0])
            rgb = np.array(rgb)       
            
            canales = [rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]]
            
            filas = rgb.shape[0]
            columnas= rgb.shape[1]
            salR = np.zeros((filas,columnas))
            salG = np.zeros((filas,columnas))
            salB = np.zeros((filas,columnas))
            salida = [salR,salG,salB]
            
            #Creacion de la nueva imagen
            filas, columnas, caneles = rgb.shape
            
            #creacion de la mascara 
             
            A = float(val)
            n = 3
            W = (n**2)*A - 1
            mask = np.ones((n,n))*-1
            pxMedioMask = int(np.round(n/2)) -1
            mask[pxMedioMask,pxMedioMask] = W
            
            #variables de salida de los canales a procesar
            salidaR = np.zeros((filas-n,columnas-n))
            salidaG = np.zeros((filas-n,columnas-n))
            salidaB = np.zeros((filas-n,columnas-n))
            salida = [salidaR,salidaG,salidaB]
            
            #procesamiento High BOOST
            
            for canal in range(3):
                for i in range(filas-n):
                    for j in range(columnas-n):
                    
                        bloqueImg = rgb[i:i+n,j:j+n,canal]
                    
                        operacion = (1/n**2)*np.sum(bloqueImg*mask)
                    
                        salida[canal][i,j] = operacion
                    
            #reconstruyendo
            imgTrans1 = np.zeros((filas-n,columnas-n,3))
            imgTrans1[:,:,0] = salida[0]
            imgTrans1[:,:,1] = salida[1]
            imgTrans1[:,:,2] = salida[2]
            
            
            imgTrans = np.zeros((filas-n,columnas-n,3))
            
            for i in range(3):
                imgTrans[:,:,i] = np.clip(imgTrans1[:,:,i], 0, 255)
            
            
            
            imgEq = np.uint8(imgTrans)
            imgEq = Image.fromarray((imgEq).astype(np.uint8))
            imgEq.save('./HB'+ str(self.contador)+ '.jpg')       
           
            self.ids.my_image.source = './HB'+ str(self.contador)+ '.jpg'
        except:
            print("error")
        
    
    def fuzz(self,filename,s1,s2,s3):
        try:
            self.contador = self.contador + 1
            
            def gbellmf(universo,a,b,c):
                rango = np.zeros(universo.shape[0])
                rango = 1 / (1 + abs((universo - c)/a)**(2*b))
                return rango
            
            def zmf(universo,a,b):
                rango = np.zeros(universo.shape)
                for i in range(universo.shape[0]):
                    if universo[i] <= a:
                        rango[i] = 1
                    elif a < universo[i] <= (a+b)/2:
                        rango[i] = 1 - 2*((universo[i]-a)/(b-a))**2
                    elif (a+b)/2 < universo[i] <= b:
                        rango[i] = 2*((universo[i]-b)/(b-a))**2
                    else:
                        rango[i] = 0
                return rango
            
            def smf(universo,a,b):
                rango = np.zeros(universo.shape)
                for i in range(universo.shape[0]):
                    if universo[i] <= a:
                        rango[i] = 0
                    elif a < universo[i] <= (a+b)/2:
                        rango[i] = 2*((universo[i]-a)/(b-a))**2
                    elif (a+b)/2 < universo[i] <= b:
                        rango[i] = 1 - 2*((universo[i]-b)/(b-a))**2
                    else:
                        rango[i] = 1
                return rango
                    
            
            # pixel = np.linspace(0, 255, 256)
            # claros = fuzz.smf(pixel, 130, 230) #red
            # grises = fuzz.gbellmf(pixel, 55, 3, 128) #blue
            # oscuros = fuzz.zmf(pixel, 25, 130) #green
            
            pixel = np.linspace(0, 255, 256)
            claros = smf(pixel, 130, 230) #red
            grises = gbellmf(pixel, 55, 3, 128) #blue
            oscuros = zmf(pixel, 25, 130) #green
            
            
            #sigletons a la salida, el usuario puede modificar 
            #defendiendo de la agresividad del histograma
            
            #grafico de salida, reglas
            salida = np.zeros(256)
            for i in range (256):
                salida [i] = ((oscuros[i]*s1)+(grises[i]*s2)+(claros[i]*s3)) / (oscuros[i]+grises[i]+claros[i])
                
            gris = Image.open(filename[0])
            gris = np.array(gris)
            
            [filas, columnas] = gris.shape
            c = np.zeros( (filas, columnas) )
            
            
            
            for i in range(filas):
                for j in range(columnas):
                    valor = int(gris[i, j])
                    c[i,j] = np.uint8(salida[valor])
            
            imgEq = np.uint8(c)
            imgEq = Image.fromarray((imgEq).astype(np.uint8))
            imgEq.save('./Fuzz'+ str(self.contador)+ '.jpg')       
           
            self.ids.my_image.source = './Fuzz'+ str(self.contador)+ '.jpg'
        except:
            print("error")

class MainApp(App):
    title = "Imagenología"
    
    def build(self):
        self.icon = "./icon.png"
        if platform == "android":
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])
        return contenedor()
    

    

if __name__ == "__main__": #condicion de plataformas que requieren esta condicion 
    MainApp().run()
    
# la unica libreria que funciono fue la de pil, puedo que pygame funciones pero no se ha probado           
    #PIL
      #https://stackoverflow.com/questions/14452824/how-can-i-save-an-image-with-pil      
      
      #pygame
      #https://stackoverflow.com/questions/34673424/how-to-get-numpy-array-of-rgb-colors-from-pygame-surface
      #-https://stackoverflow.com/questions/38557731/how-to-import-python-pil-on-android-and-kivy
      