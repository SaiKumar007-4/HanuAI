import pandas as pd
import numpy as np
import re
import ast
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score

import sys
sys.stdout.reconfigure(encoding='utf-8')

print("[*] Loading dataset...")
df = pd.read_csv('Hanuai_task2_assignment.csv')
print(f"[OK] Loaded {len(df)} rows and {len(df.columns)} columns.")

# ==============================================================================
# 1. EXPLORATORY DATA ANALYSIS (EDA) & DATA CLEANING
# ==============================================================================

# A. Inconsistency & Character Cleaning
# Clean non-standard unicode / question marks / dashes in COMPLAINT_CD_DESC
df['COMPLAINT_CD_DESC'] = df['COMPLAINT_CD_DESC'].str.replace('', '-', regex=False).str.strip()

# Clean missing values
df['CAUSAL_VERBATIM'] = df['CAUSAL_VERBATIM'].fillna('No causal verbatim provided')
df['CAUSAL_CD_DESC'] = df['CAUSAL_CD_DESC'].fillna('Unknown / Not Specified')

# B. Datetime Standardization & Feature Engineering
# Convert date columns to standard datetimes
df['Opened_Date_Clean'] = pd.to_datetime(df['Opened date'], errors='coerce')
df['BUILD_DATE_Clean'] = pd.to_datetime(df['BUILD_DATE'].str.replace('T.*', '', regex=True), errors='coerce')
df['IN_USE_DATE_Clean'] = pd.to_datetime(df['IN_USE_DATE'].dropna().str.replace('T.*', '', regex=True), errors='coerce')

# Impute single missing IN_USE_DATE with median delivery lag from BUILD_DATE
delivery_lag = (df['IN_USE_DATE_Clean'] - df['BUILD_DATE_Clean']).dt.days
median_lag = delivery_lag.median()
df['IN_USE_DATE_Clean'] = df['IN_USE_DATE_Clean'].fillna(df['BUILD_DATE_Clean'] + pd.Timedelta(days=median_lag))

# Calculate Time-to-Failure (TTF in Days)
df['Days_to_Failure'] = (df['Opened_Date_Clean'] - df['IN_USE_DATE_Clean']).dt.days
# Clean minor anomalies where opened date was recorded slightly prior to in_use (set min 0)
df['Days_to_Failure'] = df['Days_to_Failure'].apply(lambda x: max(x, 0) if pd.notnull(x) else np.nan)

# Delivery Lag (Days between factory build and customer in-use)
df['Delivery_Lag_Days'] = (df['IN_USE_DATE_Clean'] - df['BUILD_DATE_Clean']).dt.days

print("\n--- Key Statistical Summary for Numerical Features ---")
print(df[['Days_to_Failure', 'Delivery_Lag_Days']].describe().round(2))

# ==============================================================================
# 2. NLP TEXT MINING & ENTITY TAG EXTRACTION ENGINE
# ==============================================================================

print("\n[*] Running NLP text mining extraction on 1,000 incident verbatims...")

def extract_entities_for_row(row, idx):
    # Check if ground truth was already provided in initial rows (idx 0 to 4)
    if idx < 5 and pd.notnull(row['Trigger']) and row['Trigger'] != '':
        try:
            return {
                'Trigger': row['Trigger'],
                'Failure Component': row['Failure Component'],
                'Failure Condition': row['Failure Condition'],
                'Additional Context': row['Additional Context'],
                'Fix Component': row['Fix Component'],
                'Fix Condition': row['Fix Condition']
            }
        except:
            pass

    cust = str(row['CUSTOMER_VERBATIM']).lower()
    causal = str(row['CAUSAL_VERBATIM']).lower()
    corr = str(row['CORRECTION_VERBATIM']).lower()
    full_text = f"{cust} {causal} {corr}"

    # 1. TRIGGER EXTRACTION
    triggers = []
    if any(k in full_text for k in ['revers', 'reversa', 'back up', 'backup', 'parking', 'park assist']):
        triggers.append('When Reversing')
    if any(k in full_text for k in ['while driving', 'test drive', 'en carretera', 'road test']):
        triggers.append('While Driving')
    if any(k in full_text for k in ['starting', 'cranking', 'ignition', 'start up', 'when start', 'al arrancar', 'key on']):
        triggers.append('When Starting')
    if any(k in full_text for k in ['shifting', 'shift from park', 'gear change']):
        triggers.append('When Shifting')
    if any(k in full_text for k in ['connecting', 'bluetooth pairing', 'plug in', 'carplay connection', 'android auto']):
        triggers.append('When Connecting')
    if not triggers:
        triggers = ['No Additional Functionality']

    # 2. FAILURE COMPONENT EXTRACTION
    f_comps = []
    # Hierarchy and priority detection
    if any(k in cust or k in causal for k in ['screen', 'display', 'pantalla', 'touch screen', 'touchscreen', 'monitor']):
        f_comps.append('Display')
    if any(k in cust or k in causal for k in ['radio module', 'receiver module', 'head unit', 'radio assembly', 'radio receiver', 'radio']):
        if 'module' in cust or 'module' in causal:
            f_comps.append('Radio Module')
        else:
            f_comps.append('Radio')
    if any(k in cust or k in causal for k in ['audio', 'sound', 'speaker', 'chime']):
        f_comps.append('Audio Unit')
    if any(k in cust or k in causal for k in ['onstar', 'on star', 'tcicm', 'telematics']):
        f_comps.append('OnStar')
    if any(k in cust or k in causal for k in ['antenna', 'coaxial', 'coax', 'antena']):
        f_comps.append('Antenna')
    if any(k in cust or k in causal for k in ['camera', 'rear view camera', 'camara', 'vpcm', 'back up camera']):
        f_comps.append('Camera')
    if any(k in cust or k in causal for k in ['amplifier', 'amp ']):
        f_comps.append('Amplifier')
    if any(k in cust or k in causal for k in ['sd card', 'card reader']):
        f_comps.append('SD Card Reader')
    if any(k in cust or k in causal for k in ['carplay', 'android auto', 'bluetooth', 'phone']):
        f_comps.append('Phone / Connectivity')
    if any(k in cust or k in causal for k in ['usb', 'usb port', 'puerto usb']):
        f_comps.append('USB Port')
    if any(k in cust or k in causal for k in ['wiring', 'harness', 'connector', 'cable', 'terminal', 'pin ']):
        f_comps.append('Wiring / Harness')
    if any(k in cust or k in causal for k in ['cluster', 'instrument cluster', 'ipc', 'gauges']):
        f_comps.append('Instrument Cluster')
    if not f_comps:
        f_comps = ['Radio']
    f_comps = list(dict.fromkeys(f_comps))

    # 3. FAILURE CONDITION EXTRACTION
    f_conds = []
    if any(k in full_text for k in ['black screen', 'pantalla se pone negra', 'went black', 'blank screen', 'screen dark', 'dark screen', 'screen blank', 'screen is just dark', 'blacks out', 'goes blank']):
        f_conds.append('Black Screen')
    if any(k in full_text for k in ['no sound', 'no audio', 'sound cut', 'audio cut', 'loss of sound', 'no coming soung', 'without sound']):
        f_conds.append('No Sound')
    if any(k in full_text for k in ['frozen', 'freezing', 'freeze', 'locks up', 'lock up', 'reboot', 'resetting', 'restarting', 'loop']):
        f_conds.append('Freezing / Rebooting')
    if any(k in full_text for k in ['touch screen not responding', 'irresponsive', 'buttons quit working', 'knobs quit', 'slow to change', 'gets stuck', 'wont change']):
        f_conds.append('Controls Irresponsive')
    if any(k in full_text for k in ['lost communication', 'loss communication', 'no communication', 'no comm', 'ethernet bus']):
        f_conds.append('Communication Loss')
    if any(k in full_text for k in ['lost satellite signal', 'signal lost', 'channel not available', 'xm inop', 'satellite radio']):
        f_conds.append('Signal Lost')
    if any(k in full_text for k in ['internal fault', 'internal failure', 'internal malfunction']):
        f_conds.append('Internal Fault')
    if any(k in full_text for k in ['wont connect', 'cannot connect', 'connection failure', 'failed to connect', 'declines everytime']):
        f_conds.append('Connectivity Failure')
    if any(k in full_text for k in ['inop', 'inoperable', 'does not work', 'wont work', 'malfunction']):
        if not f_conds:
            f_conds.append('Inoperative')
    if not f_conds:
        f_conds = ['Malfunction']
    f_conds = list(dict.fromkeys(f_conds))

    # 4. ADDITIONAL CONTEXT EXTRACTION
    contexts = []
    if any(k in full_text for k in ['intermittent', 'intermittently', 'at times', 'sometimes', 'periodically']):
        contexts.append('Intermittent')
    if any(k in full_text for k in ['while driving', 'test drive']):
        contexts.append('While Driving')
    if any(k in full_text for k in ['at delivery', 'time of delivery', 'pre-delivery']):
        contexts.append('At Delivery')
    if any(k in full_text for k in ['customer returned', 'second time', 'came back', 'guest returned', 'repeat']):
        contexts.append('Repeat Visit')
    if not contexts:
        contexts = ['No Additional Context']
    contexts = list(dict.fromkeys(contexts))

    # 5. FIX COMPONENT EXTRACTION (FROM CORRECTION_VERBATIM)
    fix_comps = []
    if any(k in corr for k in ['radio', 'head unit', 'receiver module', 'radio receiver', 'radio assembly']):
        fix_comps.append('Radio')
    if any(k in corr for k in ['sps', 'programming', 'reprogram', 'calibration', 'calibrations']):
        fix_comps.append('SPS')
    if any(k in corr for k in ['usb', 'file transfer']):
        fix_comps.append('USB')
    if any(k in corr for k in ['antenna', 'coax', 'coaxial']):
        fix_comps.append('Antenna')
    if any(k in corr for k in ['display', 'screen', 'bezel', 'face plate']):
        fix_comps.append('Display')
    if any(k in corr for k in ['wire', 'wiring', 'harness', 'terminal', 'connector', 'pin ']):
        fix_comps.append('Wiring / Harness')
    if any(k in corr for k in ['sd card', 'card reader']):
        fix_comps.append('SD Card Reader')
    if any(k in corr for k in ['camera', 'vpcm']):
        fix_comps.append('Camera')
    if any(k in corr for k in ['amplifier', 'amp ']):
        fix_comps.append('Amplifier')
    if not fix_comps:
        fix_comps = ['Radio']
    fix_comps = list(dict.fromkeys(fix_comps))

    # 6. FIX CONDITION EXTRACTION (FROM CORRECTION_VERBATIM)
    fix_conds = []
    if any(k in corr for k in ['replaced', 'installed new', 'new radio', 'reemplace', 'remplaced', 'swapped', 'order']):
        fix_conds.append('Replaced')
    if any(k in corr for k in ['programmed', 'reprogram', 'reprogrammed', 'updated', 'calibration', 'sps', 'usb']):
        fix_conds.append('Programmed')
    if any(k in corr for k in ['repaired', 'straightened', 'bent terminal', 'reconnected', 'depinned', 'secured', 'soldered']):
        fix_conds.append('Repaired')
    if any(k in corr for k in ['reset', 'global reset', 'power cycled', 'cleared codes', 'battery disconnected']):
        fix_conds.append('Reset / Reinitialized')
    if not fix_conds:
        fix_conds = ['Replaced']
    fix_conds = list(dict.fromkeys(fix_conds))

    return {
        'Trigger': str(triggers),
        'Failure Component': str(f_comps),
        'Failure Condition': str(f_conds),
        'Additional Context': str(contexts),
        'Fix Component': str(fix_comps),
        'Fix Condition': str(fix_conds)
    }

# Execute entity extraction across all rows
extracted_data = [extract_entities_for_row(row, idx) for idx, row in df.iterrows()]
df_ext = pd.DataFrame(extracted_data)

# Update dataset columns
df['Trigger'] = df_ext['Trigger']
df['Failure Component'] = df_ext['Failure Component']
df['Failure Condition'] = df_ext['Failure Condition']
df['Additional Context'] = df_ext['Additional Context']
df['Fix Component'] = df_ext['Fix Component']
df['Fix Condition'] = df_ext['Fix Condition']

# ==============================================================================
# 3. TYPE OF ISSUE CATEGORIZATION
# ==============================================================================

def classify_issue_category(row):
    f_cond = str(row['Failure Condition'])
    f_comp = str(row['Failure Component'])
    fix_c = str(row['Fix Condition'])
    fix_comp = str(row['Fix Component'])
    causal_desc = str(row['CAUSAL_CD_DESC'])
    text = (str(row['CUSTOMER_VERBATIM']) + " " + str(row['CAUSAL_VERBATIM'])).lower()

    if 'Communication Loss' in f_cond or 'ethernet' in text or 'bus' in text or 'No/Incorrect Communication' in causal_desc:
        return 'Network & Bus Communication Issue (Ethernet/CAN)'
    elif 'Wiring / Harness' in f_comp or 'Repaired' in fix_c or 'pin ' in text or 'harness' in text or 'Wiring' in causal_desc:
        return 'Electrical & Wiring Harness Defect'
    elif 'Programmed' in fix_c and 'Replaced' not in fix_c:
        return 'Software & Calibration Anomaly'
    elif 'Black Screen' in f_cond or 'Controls Irresponsive' in f_cond or 'Display' in f_comp:
        return 'Display & Touchscreen Human-Interface Defect'
    elif 'Signal Lost' in f_cond or 'Antenna' in f_comp or 'Audio Unit' in f_comp or 'No Sound' in f_cond:
        return 'Audio & Satellite Signal Reception Fault'
    elif 'OnStar' in f_comp or 'Phone / Connectivity' in f_comp or 'Connectivity Failure' in f_cond:
        return 'Telematics & Connectivity (OnStar / WiFi / BT)'
    elif 'Internal Fault' in f_cond or 'Replaced' in fix_c:
        return 'Hardware / Component Internal Failure'
    else:
        return 'Hardware / Component Internal Failure'

df['Issue_Category'] = df.apply(classify_issue_category, axis=1)

print("\n--- Issue Category Distribution ---")
print(df['Issue_Category'].value_counts())

# ==============================================================================
# 4. CLUSTERING & TOPIC MODELING (TF-IDF + K-MEANS + LDA)
# ==============================================================================

print("\n[*] Applying TF-IDF Vectorization and K-Means Clustering on Incident Verbatims...")

# Combine narrative texts for rich NLP representation
df['Combined_Verbatim'] = (
    df['CUSTOMER_VERBATIM'].fillna('') + " " +
    df['CAUSAL_VERBATIM'].fillna('') + " " +
    df['CORRECTION_VERBATIM'].fillna('')
)

# Custom domain stop words
domain_stopwords = [
    'radio', 'vehicle', 'customer', 'tech', 'tac', 'states', 'found', 'checked', 'inspected',
    'advise', 'concern', 'time', 'case', 'gm', 'per', 'codes', 'dtc', 'dtcs', 'went', 'working',
    'codes', 'code', 'said', 'told', 'called', 'contacted', 'ordered', 'verified', 'unit',
    'replaced', 'programmed', 'perform', 'performed', 'test', 'tested', 'new', 'recheck',
    'ok', 'still', 'system', 'operation', 'completed', 'good', 'recomended', 'repaired'
]

tfidf = TfidfVectorizer(
    max_features=150,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.85
)

# Filter words
tfidf.stop_words_ = set(list(tfidf.get_stop_words()) + domain_stopwords)

X_tfidf = tfidf.fit_transform(df['Combined_Verbatim'])

# Optimal K-Means Clustering (k=5 natural failure modes)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster_ID'] = kmeans.fit_predict(X_tfidf)

# Name Clusters based on top centroid terms
cluster_names = {}
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names_out()

print("\n--- Top Keywords per Failure Cluster ---")
for i in range(k):
    top_terms = [terms[ind] for ind in order_centroids[i, :7] if terms[ind] not in domain_stopwords][:5]
    name = f"Cluster {i+1}: " + " / ".join([t.title() for t in top_terms[:3]])
    cluster_names[i] = name
    print(f"Cluster {i+1}: {', '.join(top_terms)}")

df['Cluster_Name'] = df['Cluster_ID'].map(cluster_names)

# LDA Topic Modeling (3 Latent Semantic Root Causes)
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X_tfidf)
print("\n--- Latent Dirichlet Allocation (LDA) Root Cause Topics ---")
for idx, topic in enumerate(lda.components_):
    top_features = [terms[i] for i in topic.argsort()[:-8:-1] if terms[i] not in domain_stopwords][:5]
    print(f"Topic {idx+1} (Latent Root Cause): {', '.join(top_features)}")

# ==============================================================================
# 5. EXPORT PROCESSED DATASETS (CSV & XLSX)
# ==============================================================================

# Final Submission Schema
submission_columns = [
    'Event id', 'Opened date', 'BUILD_DATE', 'IN_USE_DATE',
    'MAKE', 'MODEL', 'MODLYR', 'BUILD_PLANT_DESC', 'PLANT',
    'COMPLAINT_CD_DESC', 'CAUSAL_CD_DESC',
    'Days_to_Failure', 'Delivery_Lag_Days',
    'Trigger', 'Failure Component', 'Failure Condition',
    'Additional Context', 'Fix Component', 'Fix Condition',
    'Issue_Category', 'Cluster_ID', 'Cluster_Name',
    'CUSTOMER_VERBATIM', 'CAUSAL_VERBATIM', 'CORRECTION_VERBATIM'
]

df_export_task2 = df[submission_columns]

OUTPUT_TASK2_CSV = "task2_processed_text_mining_tags.csv"
OUTPUT_TASK2_XLSX = "task2_processed_text_mining_tags.xlsx"

df_export_task2.to_csv(OUTPUT_TASK2_CSV, index=False, encoding='utf-8')
df_export_task2.to_excel(OUTPUT_TASK2_XLSX, index=False)

print(f"\n[✓] Task 2 CSV export completed: {OUTPUT_TASK2_CSV} ({len(df_export_task2)} rows)")
print(f"[✓] Task 2 Excel export completed: {OUTPUT_TASK2_XLSX} ({len(df_export_task2)} rows)")

print("\nFirst 3 rows of processed text mining tags:")
print(df_export_task2[['Event id', 'Trigger', 'Failure Component', 'Failure Condition', 'Fix Component', 'Fix Condition', 'Issue_Category']].head(3).to_string())
