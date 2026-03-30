import torch
import soundfile as sf
import streamlit as st
import numpy as np
import pandas as pd
import io
import re
import plotly.express as px
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor, Descriptors, rdMolDescriptors
import streamlit.components.v1 as components
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ================================
# 🔧 НАСТРОЙКА СТРАНИЦЫ
# ================================
st.set_page_config(
    page_title="Caco-2 Predictor Pro", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ================================
# 🎨 СТИЛИ
# ================================
hide_style = """
    <style>
    #GithubIcon {visibility: hidden;}
    div[data-testid="stStatusWidget"] {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stButton button {
        border-radius: 20px;
        transition: all 0.3s;
    }
    
    .prediction-box {
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        text-align: center;
        font-weight: 500;
    }
    .high-perm { background: #d4edda; border: 2px solid #28a745; color: #155724; }
    .med-perm { background: #fff3cd; border: 2px solid #ffc107; color: #856404; }
    .low-perm { background: #f8d7da; border: 2px solid #dc3545; color: #721c24; }
    
    [data-testid="stChatMessageAvatarUser"]::before { content: "👤"; }
    [data-testid="stChatMessageAvatarAssistant"]::before { content: "🤖"; }
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# ================================
# 🔬 ФУНКЦИИ ДЛЯ ГЕНЕРАЦИИ ДЕСКРИПТОРОВ
# ================================
def generate_molecular_features(smiles: str) -> dict:
    """Генерация молекулярных дескрипторов для предсказания проницаемости."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    mol = Chem.AddHs(mol)
    
    features = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'MR': Descriptors.MolMR(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'RotatableBonds': Descriptors.NumRotatableBonds(mol),
        'RingCount': rdMolDescriptors.CalcNumRings(mol),
        'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'FormalCharge': Chem.GetFormalCharge(mol),
        'BalabanJ': Descriptors.BalabanJ(mol),
        'BertzCT': Descriptors.BertzCT(mol),
    }
    
    mol = Chem.RemoveHs(mol)
    return features

def render_molecule_svg(smiles: str, bond_len=50, font_size=12, pan_x=0, pan_y=0, zoom=0.9):
    """Генерация 2D SVG структуры."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "<div style='color:red; padding:20px;'>❌ Ошибка: некорректный SMILES</div>"
    
    try:
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(mol)
        
        canvas_width, canvas_height = 600, 300
        d2d = rdMolDraw2D.MolDraw2DSVG(canvas_width, canvas_height)
        opts = d2d.drawOptions()
        opts.fixedBondLength = bond_len
        opts.minFontSize = font_size
        opts.annotationFontScale = 0.8
        
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()
        
        transform = f"transform='translate({pan_x}, {pan_y}) scale({zoom})'"
        svg = svg.replace('<g>', f'<g {transform}>', 1)
        svg = re.sub(r'width=[\'"].*?[\'"]', 'width="100%"', svg)
        svg = re.sub(r'height=[\'"].*?[\'"]', 'height="100%"', svg)
        
        return f"""
        <div style='background:white; border-radius:12px; border:1px solid #ddd; 
                    overflow:auto; min-height:280px;'>
            {svg}
        </div>
        """
    except Exception as e:
        return f"<div style='color:red; padding:20px;'>Ошибка отрисовки: {str(e)}</div>"

# ================================
# 📚 ФУНКЦИИ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ
# ================================
def prepare_training_data(df):
    """Подготовка данных для обучения модели"""
    X_list = []
    y_list = []
    errors = []
    valid_smiles = []
    
    for idx, row in df.iterrows():
        smiles = row.get('SMILES', row.get('smiles', ''))
        
        if 'logPapp' in row:
            y_value = row['logPapp']
        elif 'log_Papp' in row:
            y_value = row['log_Papp']
        elif 'Papp' in row:
            papp_value = row['Papp']
            if papp_value > 0:
                y_value = np.log10(papp_value)
            else:
                errors.append(f"Строка {idx}: некорректное значение Papp ({papp_value})")
                continue
        else:
            errors.append(f"Строка {idx}: не найдена целевая переменная (нужна колонка 'Papp' или 'logPapp')")
            continue
        
        features = generate_molecular_features(smiles)
        if features:
            X_list.append(list(features.values()))
            y_list.append(y_value)
            valid_smiles.append(smiles)
        else:
            errors.append(f"Строка {idx}: не удалось обработать SMILES: {smiles}")
    
    if not X_list:
        return None, None, None, errors
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, valid_smiles, errors

def train_model(X, y, model_params=None, test_size=0.2):
    """Обучение модели RandomForest с валидацией"""
    if model_params is None:
        model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(**model_params)
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'n_samples': len(y),
        'n_train': len(y_train),
        'n_test': len(y_test)
    }
    
    X_scaled = scaler.transform(X)
    cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(y)), scoring='r2')
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()
    
    return model, scaler, metrics, (X_train, X_test, y_train, y_test, y_train_pred, y_test_pred)

# ================================
# 🧠 ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# ================================
def predict_caco2_permeability(smiles: str, model, scaler):
    """Предсказывает проницаемость через Caco-2"""
    features = generate_molecular_features(smiles)
    if not features:
        return {"error": "Не удалось распознать SMILES"}
    
    feature_names = list(features.keys())
    X = np.array([features[name] for name in feature_names]).reshape(1, -1)
    
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    if prediction > -5.0:
        category = "high"
        label = "🟢 Высокая проницаемость"
        description = "Papp > 10⁻⁵ см/с - хорошее всасывание"
    elif prediction > -6.5:
        category = "medium"
        label = "🟡 Средняя проницаемость"
        description = "10⁻⁶.⁵ < Papp < 10⁻⁵ см/с - умеренное всасывание"
    else:
        category = "low"
        label = "🔴 Низкая проницаемость"
        description = "Papp < 10⁻⁶.⁵ см/с - плохое всасывание"
    
    return {
        "log_papp": round(prediction, 3),
        "papp_value": 10**prediction,
        "papp_display": f"{10**prediction:.3e}",
        "category": category,
        "label": label,
        "description": description,
        "features": features
    }

# ================================
# 💬 ФУНКЦИИ ЧАТА
# ================================
def is_valid_smiles(text: str):
    """Проверяет, содержит ли сообщение валидный SMILES."""
    candidates = []
    
    text_clean = text.strip()
    if 2 <= len(text_clean) <= 200:
        candidates.append(text_clean)
    
    quoted = re.findall(r'["\']([CNOHSPFClBrI=\-\[\]@0-9\(\)/\\+#]+)["\']', text, re.IGNORECASE)
    candidates.extend(quoted)
    
    after_keyword = re.findall(
        r'(?:smiles|smi|mol|молекула)[:\s]+([CNOHSPFClBrI=\-\[\]@0-9\(\)/\\+#]+)', 
        text, 
        re.IGNORECASE
    )
    candidates.extend(after_keyword)
    
    for cand in candidates:
        try:
            mol = Chem.MolFromSmiles(cand)
            if mol is not None:
                return True, cand
        except:
            continue
    
    return False, ""

def get_bot_response(user_message: str, model=None, scaler=None):
    """Возвращает ответ бота."""
    msg_lower = user_message.lower()
    
    valid, smiles = is_valid_smiles(user_message)
    if valid and smiles:
        svg_html = render_molecule_svg(smiles)
        result = predict_caco2_permeability(smiles, model, scaler)
        
        if "error" in result:
            return {"type": "error", "content": f"❌ {result['error']}"}
        
        css_class = result["category"]
        
        prediction_html = f"""
        <div style="margin:10px 0;">
            <div class="prediction-box {css_class}" style="margin:0; padding:12px; font-size:0.95rem;">
                <div style="font-weight:600; margin-bottom:4px;">{result['label']}</div>
                <div style="font-size:1.3rem; font-weight:bold;">
                    Papp = {result['papp_display']} × 10⁻⁶ см/с
                </div>
                <div style="font-size:0.9rem; margin-top:4px;">
                    log Papp = {result['log_papp']}
                </div>
                <div style="font-size:0.85rem; margin-top:8px;">{result['description']}</div>
            </div>
        </div>
        """
        
        return {
            "type": "structure_prediction",
            "smiles": smiles,
            "svg": svg_html,
            "prediction": prediction_html,
            "features": result["features"]
        }
    
    if any(word in msg_lower for word in ["обуч", "train", "как обучить"]):
        return {
            "type": "text",
            "content": "📚 Как обучить модель на реальных данных:\n\nШаг 1: Подготовьте данные в формате CSV с колонками SMILES и Papp\n\nШаг 2: Загрузите файл во вкладку 'Обучение модели'\n\nШаг 3: Настройте параметры модели\n\nШаг 4: Нажмите 'Обучить модель'\n\nШаг 5: Скачайте файлы caco2_model.pkl и caco2_scaler.pkl\n\nШаг 6: Поместите файлы в папку приложения и перезапустите"
        }
    
    return {
        "type": "text",
        "content": "⚠️ Не распознал запрос. Напишите SMILES для предсказания или спросите 'как обучить модель'"
    }

# ================================
# 🚀 ИНИЦИАЛИЗАЦИЯ
# ================================
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": {"type": "text", "content": "🧪 Добро пожаловать! Я предсказываю проницаемость Caco-2 по SMILES. Перейдите на вкладку 'Обучение модели' чтобы загрузить свои данные и обучить модель!"}}
    ]

if "caco2_model" not in st.session_state:
    if os.path.exists('caco2_model.pkl') and os.path.exists('caco2_scaler.pkl'):
        try:
            st.session_state["caco2_model"] = joblib.load('caco2_model.pkl')
            st.session_state["caco2_scaler"] = joblib.load('caco2_scaler.pkl')
            st.session_state["is_real_model"] = True
        except Exception as e:
            st.session_state["caco2_model"] = None
            st.session_state["caco2_scaler"] = None
            st.session_state["is_real_model"] = False
    else:
        st.session_state["caco2_model"] = None
        st.session_state["caco2_scaler"] = None
        st.session_state["is_real_model"] = False

# ================================
# 🎯 ОСНОВНОЙ ИНТЕРФЕЙС
# ================================
st.title("🧪 Caco-2 Predictor Pro")

tab_chat, tab_train, tab_predict = st.tabs(["💬 Чат-ассистент", "📚 Обучение модели", "🔬 Предсказание"])

# ================================
# ВКЛАДКА 1: ЧАТ
# ================================
with tab_chat:
    chat_col, info_col = st.columns([1.2, 0.8])
    
    with chat_col:
        st.markdown("### 💬 Чат-помощник")
        chat_container = st.container(height=500)
        
        with chat_container:
            for msg in st.session_state["messages"][-15:]:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant" and isinstance(msg["content"], dict):
                        resp = msg["content"]
                        
                        if resp["type"] == "text":
                            st.markdown(resp["content"])
                        
                        elif resp["type"] == "structure_prediction":
                            st.caption(f"🧪 SMILES: `{resp['smiles']}`")
                            components.html(resp["svg"], height=280)
                            st.markdown(resp["prediction"], unsafe_allow_html=True)
                    else:
                        content = msg["content"]
                        if isinstance(content, dict):
                            content = content.get("content", str(content))
                        st.markdown(content)
        
        if prompt := st.chat_input("Напишите SMILES или 'как обучить модель'..."):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            
            response = get_bot_response(
                prompt, 
                st.session_state.get("caco2_model"), 
                st.session_state.get("caco2_scaler")
            )
            
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.rerun()
    
    with info_col:
        st.markdown("### 📊 Статус")
        if st.session_state["is_real_model"]:
            st.success("✅ Модель загружена")
            st.info("Используется ваша обученная модель")
        else:
            st.warning("⚠️ Модель не загружена")
            st.info("Перейдите на вкладку 'Обучение модели' чтобы обучить и загрузить модель")

# ================================
# ВКЛАДКА 2: ОБУЧЕНИЕ МОДЕЛИ
# ================================
with tab_train:
    st.markdown("## 📚 Обучение модели Caco-2")
    
    with st.expander("📄 Пример формата данных"):
        example_data = pd.DataFrame({
            'SMILES': [
                'NC1=C(N=C(N=C1N)N)N',
                'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
                'CC(=O)Oc1ccccc1C(=O)O'
            ],
            'Papp': [2.64, 15.8, 31.6]
        })
        st.dataframe(example_data, use_container_width=True)
        
        csv_example = example_data.to_csv(index=False)
        st.download_button(
            "📥 Скачать пример данных (CSV)",
            data=csv_example,
            file_name="example_training_data.csv",
            mime="text/csv"
        )
    
    st.divider()
    
    uploaded_file = st.file_uploader(
        "📁 Загрузите CSV файл с обучающими данными",
        type=['csv']
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Файл загружен: {len(df)} строк")
            
            with st.expander("📊 Превью загруженных данных"):
                st.dataframe(df.head(10), use_container_width=True)
                
                if 'SMILES' not in df.columns:
                    st.error("❌ Файл должен содержать колонку 'SMILES'")
                elif 'Papp' not in df.columns and 'logPapp' not in df.columns:
                    st.error("❌ Файл должен содержать колонку 'Papp' или 'logPapp'")
                else:
                    st.success("✅ Необходимые колонки найдены")
            
            with st.spinner("🔄 Генерация молекулярных дескрипторов..."):
                X, y, valid_smiles, errors = prepare_training_data(df)
            
            if errors:
                with st.expander("⚠️ Ошибки при обработке"):
                    for error in errors[:10]:
                        st.warning(error)
            
            if X is not None:
                st.success(f"✅ Успешно обработано {len(y)} молекул")
                
                st.markdown("### ⚙️ Настройка модели")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_estimators = st.slider("Количество деревьев", 50, 500, 200, 50)
                with col2:
                    max_depth = st.slider("Максимальная глубина", 5, 30, 15, 1)
                with col3:
                    test_size = st.slider("Размер тестовой выборки", 0.1, 0.3, 0.2, 0.05)
                
                if st.button("🚀 Обучить модель", type="primary"):
                    with st.spinner("🔄 Обучение модели..."):
                        model_params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': 5,
                            'min_samples_leaf': 2,
                            'random_state': 42
                        }
                        
                        model, scaler, metrics, _ = train_model(X, y, model_params, test_size)
                        
                        st.markdown("### 📊 Результаты обучения")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² (тест)", f"{metrics['test_r2']:.3f}")
                        with col2:
                            st.metric("RMSE (тест)", f"{metrics['test_rmse']:.3f}")
                        with col3:
                            st.metric("CV R²", f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
                        
                        st.markdown("### 💾 Скачать модель")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            model_buffer = io.BytesIO()
                            joblib.dump(model, model_buffer)
                            model_buffer.seek(0)
                            
                            st.download_button(
                                "📥 Скачать модель (caco2_model.pkl)",
                                data=model_buffer,
                                file_name="caco2_model.pkl",
                                mime="application/octet-stream"
                            )
                        
                        with col2:
                            scaler_buffer = io.BytesIO()
                            joblib.dump(scaler, scaler_buffer)
                            scaler_buffer.seek(0)
                            
                            st.download_button(
                                "📥 Скачать scaler (caco2_scaler.pkl)",
                                data=scaler_buffer,
                                file_name="caco2_scaler.pkl",
                                mime="application/octet-stream"
                            )
                        
                        st.info("📌 Поместите скачанные файлы в папку с приложением и перезапустите")
            else:
                st.error("❌ Не удалось подготовить данные для обучения")
                
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла: {str(e)}")

# ================================
# ВКЛАДКА 3: ПРЕДСКАЗАНИЕ
# ================================
with tab_predict:
    st.markdown("## 🔬 Предсказание проницаемости Caco-2")
    
    if st.session_state["is_real_model"]:
        st.success("✅ Используется ваша обученная модель")
    else:
        st.warning("⚠️ Модель не загружена. Перейдите на вкладку 'Обучение модели' чтобы обучить и загрузить модель.")
    
    smiles_input = st.text_area(
        "Введите SMILES структуру:",
        placeholder="Например: NC1=C(N=C(N=C1N)N)N (рибавирин)",
        height=100
    )
    
    if smiles_input:
        if st.button("🔮 Предсказать", type="primary"):
            if st.session_state["caco2_model"] and st.session_state["caco2_scaler"]:
                with st.spinner("🔄 Анализ молекулы..."):
                    result = predict_caco2_permeability(
                        smiles_input,
                        st.session_state["caco2_model"],
                        st.session_state["caco2_scaler"]
                    )
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.subheader("🔬 2D Структура")
                        svg_code = render_molecule_svg(smiles_input)
                        components.html(svg_code, height=350)
                        
                        st.subheader("📊 Результаты предсказания")
                        
                        css_class = result["category"]
                        st.markdown(f"""
                        <div class="prediction-box {css_class}">
                            <div style="font-size:1.4rem; margin-bottom:8px;">{result['label']}</div>
                            <div style="font-size:2rem; font-weight:bold;">
                                Papp = {result['papp_display']} × 10⁻⁶ см/с
                            </div>
                            <div style="font-size:1.2rem; margin-top:8px;">
                                log Papp = {result['log_papp']}
                            </div>
                            <div style="margin-top:12px; font-size:0.95rem;">{result['description']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📋 Молекулярные дескрипторы"):
                            df_features = pd.DataFrame([
                                {"Дескриптор": k, "Значение": f"{v:.3f}" if isinstance(v, float) else v}
                                for k, v in result["features"].items()
                            ])
                            st.dataframe(df_features, use_container_width=True, hide_index=True)
                        
                        export_data = {
                            "SMILES": smiles_input,
                            "log_Papp": result["log_papp"],
                            "Papp": result["papp_value"],
                            "category": result["category"]
                        }
                        df_export = pd.DataFrame([export_data])
                        csv = df_export.to_csv(index=False)
                        
                        st.download_button(
                            "📥 Скачать результат (CSV)",
                            data=csv,
                            file_name=f"prediction_{smiles_input[:20]}.csv",
                            mime="text/csv"
                        )
            else:
                st.error("❌ Модель не загружена. Перейдите на вкладку 'Обучение модели' чтобы обучить и загрузить модель.")
