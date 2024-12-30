from colorama import init, Fore, Back, Style
from time import strftime, localtime
import os
init()

class Logger:
    def __init__(self, filename:str):
        self.fn = filename
        os.makedirs(os.path.join(".","logs"),exist_ok=True)
        self.log_name = f"{strftime('%d_%m_%Y-%H_%M_%S', localtime())}.log"
    
    def __log(self,level:str, message:str, color):
        if level != "ANNOUNCEMENT": 
            print(f"{color}[{level.capitalize()}]{Style.RESET_ALL}: {Style.DIM}{self.fn}{Style.RESET_ALL} - {message}")
        else:
            print(f"{message}")

        with open(os.path.join("logs",self.log_name),"a") as f:
            f.write(f"\n[{level.capitalize()}]: {self.fn} - {message}")


    def critical(self,message:str):
        self.__log("CRITICAL",message,Back.MAGENTA)

    def error(self, message:str):
        self.__log("ERROR",message, Back.RED)

    def warn(self, message:str):
        self.__log("WARNING",message, Back.YELLOW)

    def info(self,message:str):
        self.__log("INFO", message, Fore.CYAN)

    def debug(self,message:str):
        self.__log("DEBUG",message, Style.DIM)

    def announcement(self,message:str):
        self.__log("ANNOUNCEMENT",message, Fore.WHITE)
        