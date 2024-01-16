import joblib

# Carregar o modelo
model = joblib.load('rf_t5.pickle')

# Salvar novamente o modelo (para atualizá-lo)
joblib.dump(model, 'rf_t5.pickle')
