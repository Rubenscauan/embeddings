import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

class FakeNewsDetector:
    """
    Sistema para detectar notícias falsas usando duas abordagens:
    1. Fine-tuning de BERT (modelo transformer)
    2. Embeddings + Classificador tradicional
    """

    def __init__(self, model_name='neuralmind/bert-base-portuguese-cased'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")

    def load_data(self, real_news_dir, fake_news_dir):
        """Carrega os datasets de notícias reais e falsas"""
        texts = []
        labels = []

        # Carregar notícias reais (label = 1)
        real_path = Path(real_news_dir)
        for file in real_path.glob('*.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1)

        # Carregar notícias falsas (label = 0)
        fake_path = Path(fake_news_dir)
        for file in fake_path.glob('*.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(0)

        print(f"Carregados {len(texts)} textos:")
        print(f"  - Notícias reais: {sum(labels)}")
        print(f"  - Notícias falsas: {len(labels) - sum(labels)}")

        return texts, labels

    # ========== ABORDAGEM 1: Fine-tuning de BERT ==========

    def train_bert_classifier(self, texts, labels, output_dir='./fake_news_bert', epochs=3):
        """Treina um classificador BERT fine-tunado"""
        print("\n=== Treinando Classificador BERT ===")

        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Carregar tokenizer e modelo
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        ).to(self.device)

        # Tokenizar dados
        train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512)

        # Criar datasets PyTorch
        train_dataset = NewsDataset(train_encodings, y_train)
        test_dataset = NewsDataset(test_encodings, y_test)

        # Configurar treinamento
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        # Treinar
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        trainer.train()

        # Avaliar
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)

        print("\n=== Resultados BERT ===")
        print(f"Acurácia: {accuracy_score(y_test, preds):.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, preds, target_names=['Falsa', 'Real']))

        # Salvar modelo
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        return model, tokenizer, accuracy_score(y_test, preds)

    # ========== ABORDAGEM 2: Embeddings + Classificador ==========

    def train_embedding_classifier(self, texts, labels,
                                   embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
                                   classifier_type='rf'):
        """Treina um classificador usando embeddings prontos"""
        print("\n=== Treinando Classificador com Embeddings ===")

        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Gerar embeddings
        print("Gerando embeddings...")
        encoder = SentenceTransformer(embedding_model)

        train_embeddings = encoder.encode(X_train, show_progress_bar=True)
        test_embeddings = encoder.encode(X_test, show_progress_bar=True)

        # Treinar classificador
        if classifier_type == 'rf':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif classifier_type == 'svm':
            classifier = SVC(kernel='rbf', C=1.0, random_state=42)
        else:
            raise ValueError("classifier_type deve ser 'rf' ou 'svm'")

        print(f"Treinando {classifier_type.upper()}...")
        classifier.fit(train_embeddings, y_train)

        # Avaliar
        y_pred = classifier.predict(test_embeddings)

        print("\n=== Resultados Embeddings + Classificador ===")
        print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=['Falsa', 'Real']))

        # Salvar modelos
        with open(f'embedding_classifier_{classifier_type}.pkl', 'wb') as f:
            pickle.dump(classifier, f)

        return encoder, classifier, accuracy_score(y_test, y_pred)

    # ========== PREDIÇÃO ==========

    def predict_bert(self, text, model_dir='./fake_news_bert'):
        """Prediz se uma notícia é falsa usando o modelo BERT"""
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)

        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                           padding=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        return {
            'label': 'Real' if pred == 1 else 'Falsa',
            'confidence': probs[0][pred].item(),
            'prob_fake': probs[0][0].item(),
            'prob_real': probs[0][1].item()
        }

    def predict_embedding(self, text, encoder, classifier):
        """Prediz se uma notícia é falsa usando embeddings"""
        embedding = encoder.encode([text])
        pred = classifier.predict(embedding)[0]

        # Tentar obter probabilidades se disponível
        if hasattr(classifier, 'predict_proba'):
            probs = classifier.predict_proba(embedding)[0]
            return {
                'label': 'Real' if pred == 1 else 'Falsa',
                'confidence': max(probs),
                'prob_fake': probs[0],
                'prob_real': probs[1]
            }
        else:
            return {
                'label': 'Real' if pred == 1 else 'Falsa',
                'confidence': None
            }


class NewsDataset(torch.utils.data.Dataset):
    """Dataset personalizado para PyTorch"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ========== EXEMPLO DE USO ==========

if __name__ == "__main__":
    # Inicializar detector
    detector = FakeNewsDetector()

    # Caminhos dos seus datasets
    REAL_NEWS_DIR = './/full_texts//fake'
    FAKE_NEWS_DIR = './/full_texts//true'

    # Carregar dados
    texts, labels = detector.load_data(REAL_NEWS_DIR, FAKE_NEWS_DIR)

    # OPÇÃO 1: Treinar BERT (mais preciso, mas mais lento)
    print("\n" + "=" * 60)
    print("TREINANDO MODELO BERT")
    print("=" * 60)
    bert_model, bert_tokenizer, bert_acc = detector.train_bert_classifier(
        texts, labels, epochs=3
    )

    # OPÇÃO 2: Treinar com Embeddings (mais rápido, bom resultado)
    print("\n" + "=" * 60)
    print("TREINANDO MODELO COM EMBEDDINGS")
    print("=" * 60)
    encoder, classifier, emb_acc = detector.train_embedding_classifier(
        texts, labels, classifier_type='rf'
    )

    # Testar predição
    print("\n" + "=" * 60)
    print("TESTANDO PREDIÇÕES")
    print("=" * 60)

    test_text = """
    Cientistas descobriram cura milagrosa para todas as doenças usando 
    apenas água e limão. Médicos não querem que você saiba disso!
    """

    print("\nTexto de teste:")
    print(test_text)

    print("\n--- Predição BERT ---")
    result_bert = detector.predict_bert(test_text)
    print(f"Resultado: {result_bert['label']}")
    print(f"Confiança: {result_bert['confidence']:.2%}")
    print(f"Prob. Falsa: {result_bert['prob_fake']:.2%}")
    print(f"Prob. Real: {result_bert['prob_real']:.2%}")

    print("\n--- Predição Embeddings ---")
    result_emb = detector.predict_embedding(test_text, encoder, classifier)
    print(f"Resultado: {result_emb['label']}")
    if result_emb['confidence']:
        print(f"Confiança: {result_emb['confidence']:.2%}")
        print(f"Prob. Falsa: {result_emb['prob_fake']:.2%}")
        print(f"Prob. Real: {result_emb['prob_real']:.2%}")