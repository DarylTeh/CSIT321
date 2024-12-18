from tkinter import *

mainMenuTitle = "Main Menu"
predefinedHGTitle = "Predefined Hand Gestures Interface"
customHGTitle = "Custom Hand Gestures Interface"
titleSize = "50"
keyboardTitle = "Keyboard"
mouseTitle = "Mouse"

uiPath = ""

# class root(Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Main Menu")
#         self.geometry("500x500")
        
#         mainFrame = Frame(self)
        
#         self.frames = {}
        
#         for page in (mainMenuUI, predefinedHandGesturesUI, customHandGesturesUI):
#             frame = page(mainFrame, self)
#             self.frames[page] = frame
#             frame.grid(row=0, column=0, sticky=(W, E, N, S))
    
#     def navigateTo(self, page):
#         frame = self.frames[page]
#         frame.tkraise()
   

# class mainMenuUI(Frame):
    
#     def __init__(self, parent, controller):
#         super().__init__()
#         self.controller = controller
        
#         title = Label(self, text=mainMenuTitle, font=titleSize)
#         title.pack()
        
#         predefinedHandGesturesBtn = buildButton(root, "Predefined Hand Gestures", lambda: controller.navigateTo(predefinedHandGesturesUI))
#         predefinedHandGesturesBtn.pack(padx=20, pady=20)
        
#         customHandGesturesBtn = buildButton(root, "Custom Hand Gestures", lambda: controller.navigateTo(customHandGesturesUI))
#         customHandGesturesBtn.pack(padx=20, pady=20)
        
#         testingBtn = buildButton(root, "Test Hand Gestures", lambda: controller.navigateTo(testingHGUI))
#         testingBtn.pack(padx=20, pady=20)
        
#         startGameBtn = buildButton(root, "Start Game", lambda: controller.navigateTo())
#         startGameBtn.pack(padx=20, pady=20)
    
# class predefinedHandGesturesUI(Frame):
    
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller
        
#         title = Label(self, text=predefinedHGTitle, font=titleSize)
#         title.pack()
        
#         # keyboardBtn = buildButton(self, "Keyboard", )
#         # mouseBtn = buildButton(self, "Mouse", )
#         # doneBtn = buildButton(self, "Done", , "green")
        
#         # keyboardBtn.pack(padx=20, pady=20)
#         # mouseBtn.pack(padx=20, pady=20)
#         # doneBtn.pack(padx=20, pady=20)
        

# class customHandGesturesUI(Frame):
    
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller
        
#         title = Label(self, text=customHGTitle, font=titleSize)
#         title.pack()
        
#         # keyboardBtn = buildButton(self, "Keyboard", )
#         # mouseBtn = buildButton(self, "Mouse", )
#         # doneBtn = buildButton(self, "Done", , "green")
        
#         # keyboardBtn.pack(padx=20, pady=20)
#         # mouseBtn.pack(padx=20, pady=20)
#         # doneBtn.pack(padx=20, pady=20)  
        
# class testingHGUI(Frame):
    
#     def __init__(self, parent, controller):
#         super().__init__(parent)
#         self.controller = controller
        
#         title = Label(self, text=customHGTitle, font=titleSize)
#         title.pack()
        
#         # keyboardBtn = buildButton(self, "Keyboard", )
#         # mouseBtn = buildButton(self, "Mouse", )
#         # doneBtn = buildButton(self, "Done", , "green")
        
#         # keyboardBtn.pack(padx=20, pady=20)
#         # mouseBtn.pack(padx=20, pady=20)
#         # doneBtn.pack(padx=20, pady=20)    

    
# def buildButton(frame, text, actionFunc):
    
#     button = Button(
#         frame,
#         text=text,
#         command= actionFunc(),
#         activebackground="blue",
#         activeforeground="white",
#         anchor="center",
#         bd=3,
#         bg="lightgray",
#         cursor="hand2",
#         foreground="black",
#         fg="black",
#         font=("Arial", 12),
#         height=2,
#         highlightbackground="black",
#         highlightcolor="green",
#         highlightthickness=2,
#         justify="center",
#         overrelief="raised",
#         padx=10,
#         pady=5,
#         width=15,
#         wraplength=100
#     )
    
#     return button

# def buildButtonWithColor(frame, text, actionFunc, color):
    
#     button = Button(
#         frame,
#         text=text,
#         command= actionFunc(),
#         activebackground=color,
#         activeforeground="white",
#         anchor="center",
#         bd=3,
#         bg="lightgray",
#         cursor="hand2",
#         foreground="black",
#         fg="black",
#         font=("Arial", 12),
#         height=2,
#         highlightbackground="black",
#         highlightcolor="green",
#         highlightthickness=2,
#         justify="center",
#         overrelief="raised",
#         padx=10,
#         pady=5,
#         width=15,
#         wraplength=100
#     )
    
#     return button
    
    