import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from collections import Counter

# ── Load & clean ──────────────────────────────────────────────────────────────
df_raw = pd.read_excel('/mnt/user-data/uploads/NWU_student_smoking_Analysis.xlsx', sheet_name='Data')

# Keep only real survey rows (drop pivot/summary rows that leaked in)
df = df_raw[pd.to_numeric(df_raw['ID'], errors='coerce').notna()].copy()
df = df[df['How old are you?'].apply(lambda x: str(x).isdigit() or (isinstance(x, (int,float)) and 15 < x < 60))].copy()
df['How old are you?'] = pd.to_numeric(df['How old are you?'], errors='coerce')
df['Do you smoke?'] = pd.to_numeric(df['Do you smoke?'], errors='coerce')
df = df[df['Do you smoke?'].isin([0,1])].copy()

# Gender mapping
df['Gender'] = df['What is your gender?'].map({1: 'Male', 2: 'Female', 3: 'Other'})
df = df[df['Gender'].isin(['Male','Female','Other'])].copy()

# Faculty clean
faculty_map = {
    'Faculty of Natural and Agricultural Sciences': 'Natural & Agri Sciences',
    'Faculty of Economics and Management Sciences': 'Economics & Management',
    'Faculty of Economics and Management Sciences\xa0': 'Economics & Management',
    'Faculty of Health Sciences': 'Health Sciences',
    'Faculty of Law': 'Law',
    'Faculty of Education': 'Education',
    'Faculty of Humanities': 'Humanities',
    'Faculty of Theology': 'Theology',
    'Faculty of Engineering': 'Engineering',
}
df['Faculty_Clean'] = df['Faculty'].map(faculty_map)
df = df[df['Faculty_Clean'].notna()].copy()

smokers = df[df['Do you smoke?'] == 1]
non_smokers = df[df['Do you smoke?'] == 0]

# ── Palette 
SMOKE_COLOR   = "#E05252"
NOSMOKE_COLOR = "#4A90D9"
ACCENT        = "#F4A623"
BG            = "#F9FAFB"
DARK          = "#2C3E50"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': BG,
    'figure.facecolor': 'white',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.titlepad': 14,
})

fig = plt.figure(figsize=(20, 26))
fig.suptitle("NWU Student Smoking Behaviour Analysis", fontsize=22, fontweight='bold',
             color=DARK, y=0.98)
fig.text(0.5, 0.965, f"Survey of {len(df)} students across multiple faculties  |  NWU, 2025",
         ha='center', fontsize=12, color='gray')

gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.38,
                      left=0.07, right=0.97, top=0.94, bottom=0.04)

# ── 1. Smokers vs Non-Smokers (donut)
ax1 = fig.add_subplot(gs[0, 0])
counts = [len(non_smokers), len(smokers)]
labels = ['Non-Smokers', 'Smokers']
colors = [NOSMOKE_COLOR, SMOKE_COLOR]
wedges, texts, autotexts = ax1.pie(
    counts, labels=None, colors=colors, autopct='%1.0f%%',
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
)
for at in autotexts:
    at.set_fontsize(13); at.set_fontweight('bold'); at.set_color('white')
ax1.legend(wedges, [f'{l} ({c})' for l, c in zip(labels, counts)],
           loc='lower center', bbox_to_anchor=(0.5, -0.12), fontsize=10, frameon=False)
ax1.set_title("Smoker Prevalence")

# ── 2. Gender breakdown
ax2 = fig.add_subplot(gs[0, 1])
gender_smoke = df.groupby(['Gender', df['Do you smoke?'].map({1:'Smoker',0:'Non-Smoker'})]).size().unstack(fill_value=0)
gender_smoke = gender_smoke[['Non-Smoker','Smoker']] if 'Non-Smoker' in gender_smoke.columns else gender_smoke
x = np.arange(len(gender_smoke))
w = 0.35
bars1 = ax2.bar(x - w/2, gender_smoke.get('Non-Smoker', 0), w, color=NOSMOKE_COLOR, label='Non-Smoker', edgecolor='white')
bars2 = ax2.bar(x + w/2, gender_smoke.get('Smoker', 0), w, color=SMOKE_COLOR, label='Smoker', edgecolor='white')
ax2.set_xticks(x); ax2.set_xticklabels(gender_smoke.index, fontsize=11)
ax2.set_ylabel('Count'); ax2.legend(frameon=False, fontsize=10)
ax2.set_title("Smoking by Gender")
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    if h > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.1, str(int(h)), ha='center', va='bottom', fontsize=9)

# ── 3. Age distribution 
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(non_smokers['How old are you?'].dropna(), bins=range(17,36), alpha=0.7,
         color=NOSMOKE_COLOR, label='Non-Smoker', edgecolor='white')
ax3.hist(smokers['How old are you?'].dropna(), bins=range(17,36), alpha=0.7,
         color=SMOKE_COLOR, label='Smoker', edgecolor='white')
ax3.set_xlabel('Age'); ax3.set_ylabel('Count')
ax3.legend(frameon=False, fontsize=10)
ax3.set_title("Age Distribution by Smoking Status")

# ── 4. Smoking frequency
ax4 = fig.add_subplot(gs[1, 0])
freq = smokers['How Often?'].value_counts()
bar_colors = [SMOKE_COLOR, ACCENT, "#C0392B"][:len(freq)]
bars = ax4.barh(freq.index, freq.values, color=bar_colors, edgecolor='white')
ax4.set_xlabel('Number of Students')
ax4.set_title("Smoking Frequency")
for bar in bars:
    w = bar.get_width()
    ax4.text(w + 0.05, bar.get_y() + bar.get_height()/2, str(int(w)), va='center', fontsize=10)

# ── 5. Products used 
ax5 = fig.add_subplot(gs[1, 1])
all_products = []
for entry in smokers['Type of product'].dropna():
    all_products.extend([p.strip().rstrip(';') for p in entry.split(';') if p.strip()])
prod_counts = Counter(all_products)
prod_df = pd.Series(prod_counts).sort_values()
colors_p = [SMOKE_COLOR if 'Marijuana' in p else NOSMOKE_COLOR if 'Vape' in p else ACCENT for p in prod_df.index]
ax5.barh(prod_df.index, prod_df.values, color=colors_p, edgecolor='white')
ax5.set_xlabel('Mentions')
ax5.set_title("Products Used by Smokers")
for i, v in enumerate(prod_df.values):
    ax5.text(v + 0.05, i, str(v), va='center', fontsize=10)

# ── 6. Gateway drug 
ax6 = fig.add_subplot(gs[1, 2])
gw = smokers['What was your gateway drug?'].value_counts()
wedges2, texts2, autotexts2 = ax6.pie(
    gw.values, labels=None, autopct='%1.0f%%', startangle=140,
    colors=[SMOKE_COLOR, ACCENT, NOSMOKE_COLOR, "#8E44AD"][:len(gw)],
    wedgeprops=dict(edgecolor='white', linewidth=2), pctdistance=0.75
)
for at in autotexts2:
    at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')
ax6.legend(wedges2, gw.index, loc='lower center', bbox_to_anchor=(0.5, -0.12), fontsize=10, frameon=False)
ax6.set_title("Gateway Substance")

# ── 7. Why they started 
ax7 = fig.add_subplot(gs[2, 0])
all_reasons = []
for entry in smokers['Why did you start'].dropna():
    all_reasons.extend([r.strip().rstrip(';') for r in entry.split(';') if r.strip()])
reason_counts = Counter(all_reasons)
reason_df = pd.Series(reason_counts).sort_values()
ax7.barh(reason_df.index, reason_df.values, color=SMOKE_COLOR, edgecolor='white', alpha=0.85)
ax7.set_xlabel('Mentions')
ax7.set_title("Why Students Started Smoking")
for i, v in enumerate(reason_df.values):
    ax7.text(v + 0.05, i, str(v), va='center', fontsize=9)
ax7.tick_params(axis='y', labelsize=8)

# ── 8. Why non-smokers don't smoke
ax8 = fig.add_subplot(gs[2, 1])
all_nodreasons = []
for entry in non_smokers["Why don't you smoke?"].dropna():
    all_nodreasons.extend([r.strip().rstrip(';') for r in entry.split(';') if r.strip()])
noreason_counts = Counter(all_nodreasons)
noreason_df = pd.Series(noreason_counts).sort_values()
ax8.barh(noreason_df.index, noreason_df.values, color=NOSMOKE_COLOR, edgecolor='white', alpha=0.85)
ax8.set_xlabel('Mentions')
ax8.set_title("Why Non-Smokers Don't Smoke")
for i, v in enumerate(noreason_df.values):
    ax8.text(v + 0.05, i, str(v), va='center', fontsize=9)
ax8.tick_params(axis='y', labelsize=8)

# ── 9. Health awareness heatmap
ax9 = fig.add_subplot(gs[2, 2])
health_cols = ['Fertility Issues', 'Lung, Mouth and Throat Cancer',
               'Erectile Dysfunction', 'Cardiovascular Disease',
               'Respiratory Illnesses\xa0', 'Type 2 Diabetes']
short_labels = ['Fertility\nIssues', 'Lung/Mouth\nCancer', 'Erectile\nDysfunction',
                'Cardio-\nvascular', 'Respiratory\nIllness', 'Type 2\nDiabetes']
for col in health_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
awareness = df.groupby(df['Do you smoke?'].map({1:'Smoker',0:'Non-Smoker'}))[health_cols].mean() * 100
awareness.columns = short_labels
sns.heatmap(awareness, annot=True, fmt='.0f', cmap='RdYlBu_r', ax=ax9,
            linewidths=0.5, cbar_kws={'label': '% Aware', 'shrink': 0.8},
            annot_kws={'size': 9})
ax9.set_title("Health Risk Awareness (%)")
ax9.set_ylabel('')
ax9.tick_params(axis='x', labelsize=8)

# ── 10. Faculty smoking rates 
ax10 = fig.add_subplot(gs[3, :2])
fac_smoke = df.groupby('Faculty_Clean')['Do you smoke?'].agg(['sum','count'])
fac_smoke['rate'] = fac_smoke['sum'] / fac_smoke['count'] * 100
fac_smoke = fac_smoke.sort_values('rate', ascending=True)
bar_cols = [SMOKE_COLOR if r >= 40 else ACCENT if r >= 25 else NOSMOKE_COLOR for r in fac_smoke['rate']]
bars10 = ax10.barh(fac_smoke.index, fac_smoke['rate'], color=bar_cols, edgecolor='white')
ax10.set_xlabel('Smoking Rate (%)')
ax10.set_title("Smoking Rate by Faculty")
ax10.axvline(fac_smoke['rate'].mean(), color=DARK, linestyle='--', alpha=0.5, linewidth=1.5)
ax10.text(fac_smoke['rate'].mean() + 0.5, -0.6, f"Avg {fac_smoke['rate'].mean():.0f}%", fontsize=9, color=DARK)
for bar in bars10:
    w = bar.get_width()
    ax10.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.0f}%', va='center', fontsize=9)

p1 = mpatches.Patch(color=SMOKE_COLOR, label='High (≥40%)')
p2 = mpatches.Patch(color=ACCENT, label='Medium (25-39%)')
p3 = mpatches.Patch(color=NOSMOKE_COLOR, label='Low (<25%)')
ax10.legend(handles=[p1,p2,p3], frameon=False, fontsize=9, loc='lower right')

# ── 11. Duration smoked
ax11 = fig.add_subplot(gs[3, 2])
dur_order = ['0 - 6 months', '6 months - 1 year', '1 - 2 years', 'More than 2 years']
dur = smokers['For how long have you been\xa0smoking?'].value_counts()
dur = dur.reindex([d for d in dur_order if d in dur.index])
ax11.bar(range(len(dur)), dur.values, color=[ACCENT, SMOKE_COLOR, "#C0392B", DARK][:len(dur)], edgecolor='white')
ax11.set_xticks(range(len(dur)))
ax11.set_xticklabels([d.replace(' - ', '\n') for d in dur.index], fontsize=8)
ax11.set_ylabel('Students')
ax11.set_title("How Long Have They Been Smoking?")
for i, v in enumerate(dur.values):
    ax11.text(i, v + 0.05, str(v), ha='center', fontsize=10, fontweight='bold')

plt.savefig('/mnt/user-data/outputs/NWU_Smoking_Analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print("Saved!")
