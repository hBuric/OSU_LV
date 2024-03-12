print("Unesite ocijenu od 0.0 do 1.0:")

try:
    ocjena=float(input())
    if(ocjena<0 or ocjena>1.0):
        print("Uneseni broj nije u intervalu")    
    elif(ocjena>=0.9):
        print("Ocjena je A")
    elif(ocjena>=0.8):
        print("Ocjena je B")
    elif(ocjena>=0.7):
        print("Ocjena je C")
    elif(ocjena>=0.6):
        print("Ocjena je D")
    elif(ocjena<0.6):
        print("Ocjena je F")
except:
    print("Unesena vrijednost nije broj")