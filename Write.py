class WriteFile:
    
    def write(self, name, value):
        
        with open(f"ANN/Files/{name}.txt", 'w') as f:
            f.write(str(value))

