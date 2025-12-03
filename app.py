import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import date

# --- CONFIGURATION GÉNÉRALE ---
st.set_page_config(
    page_title="Romain Chalard - Portfolio Structuration",
    page_icon="📈",
    layout="wide"
)

# --- CSS PERSONNALISÉ (Pour un look un peu plus "Finance") ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #0E1117; font-weight: 700;}
    .sub-header {font-size: 1.5rem; color: #4F8BF9; font-weight: 600;}
    .highlight {background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 5px solid #4F8BF9;}
</style>
""", unsafe_allow_html=True)

# --- DONNÉES DU CV (Hardcodées pour la simplicité) ---
CONTACT_INFO = {
    "name": "Romain Chalard",
    "tagline": "Étudiant en Ingénierie Financière | Futur Analyste Structuration",
    "phone": "+33 7 81 78 79 71",
    "email": "romain.chalard@student.junia.com",
    "location": "Paris, France",
    "linkedin": "https://linkedin.com/in/votre-profil", # À modifier
    "github": "https://github.com/votre-pseudo"        # À modifier
}

# --- FONCTIONS UTILITAIRES (Black-Scholes) ---
def black_scholes(S, K, T, r, sigma, option_type="Call"):
    """Calcule le prix et les Grecques d'une option Européenne."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100 
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "Call" else -d2)) / 365
    
    return price, delta, gamma, vega, theta, rho

# --- SIDEBAR (Barre latérale) ---
with st.sidebar:
    # Placeholder pour la photo (si vous n'avez pas l'image, cela affichera un gris)
    try:
        st.image("profile_pic.jpg", use_container_width=True)
    except:
        st.warning("Ajoutez 'profile_pic.jpg' dans le dossier")
        
    st.title(CONTACT_INFO["name"])
    st.write(CONTACT_INFO["location"])
    
    st.markdown("---")
    
    # Bouton téléchargement CV
    try:
        with open("cv_romain_chalard.pdf", "rb") as pdf_file:
            st.download_button(
                label="📄 Télécharger mon CV",
                data=pdf_file,
                file_name="CV_Romain_Chalard.pdf",
                mime="application/pdf"
            )
    except:
        st.info("Le fichier PDF du CV n'est pas encore chargé.")

    st.markdown("### Contact")
    st.write(f"📧 {CONTACT_INFO['email']}")
    st.write(f"📱 {CONTACT_INFO['phone']}")
    st.markdown(f"[LinkedIn]({CONTACT_INFO['linkedin']}) | [GitHub]({CONTACT_INFO['github']})")
    
    st.markdown("---")
    st.caption("Développé en Python & Streamlit")

# --- CONTENU PRINCIPAL ---

# Titre Principal
st.markdown(f'<div class="main-header">Portfolio Technique & Financier</div>', unsafe_allow_html=True)
st.markdown(f"**{CONTACT_INFO['tagline']}**")

# Onglets de navigation
tab_about, tab_skills, tab_tech = st.tabs(["👤 À Propos & Ambitions", "💼 Compétences & Expériences", "💻 Labo Structuration (Code)"])

# --- TAB 1 : À PROPOS & AMBITIONS ---
with tab_about:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Mon Objectif : La Structuration")
        st.info("""
        **Recherche de stage (6 mois) à partir de Juin 2026**
        
        Actuellement en cycle ingénieur à **Junia HEI** (Lille), je construis mon parcours autour d'une double compétence : 
        l'ingénierie financière (Maths/Code) et l'agilité commerciale. 
        
        Je rejoindrai l'**University of Florida** en Janvier 2026 pour me spécialiser en Finance de Marché.
        """)
        
        st.markdown("### 📈 Intérêt Personnel pour les Marchés")
        st.write("""
        Au-delà de ma formation académique, je suis un investisseur particulier actif. Cette pratique quotidienne me permet de :
        * **Confronter la théorie à la réalité :** J'applique l'analyse fondamentale (ratios, bilans) et technique pour gérer mon propre portefeuille.
        * **Suivre la Macroéconomie :** Je surveille l'impact des politiques des banques centrales sur les différentes classes d'actifs.
        * **Gérer le Risque :** J'apprends à maîtriser la psychologie de marché et le money management en conditions réelles.
        """)

    with col2:
        st.markdown("### 🎓 Formation Clé")
        st.markdown("""
        **2026 (Jan-Mai)** 🇺🇸 **University of Florida** *Finance de Marché & Supply Chain*
        
        **2024 - Présent** 🇫🇷 **Junia HEI, Lille** *Ingénierie Financière* *(Maths, VBA, Analyse Financière)*
        
        **2019 - 2022** 🇺🇸 **Academica High School** *Dual Diploma (US High School Diploma)*
        """)

# --- TAB 2 : COMPÉTENCES & EXPÉRIENCES ---
with tab_skills:
    st.markdown("### 🛠 Compétences démontrées par l'expérience")
    st.markdown("Je ne liste pas simplement des mots-clés, je les applique concrètement.")
    
    # [cite_start]On utilise les données extraites du CV [cite: 22, 28, 19, 13]
    skills_data = [
        {"Compétence": "Modélisation Mathématique", "Contexte": "Stage Sodexo Bateaux Parisiens", "Réalisation": "Conception d'un modèle complet d'émissions de CO2 sur Excel/VBA pour toute la flotte."},
        {"Compétence": "Leadership & Budget", "Contexte": "Président Club Oenologie", "Réalisation": "Gestion d'un budget de 6k€, management de 20 membres, négociation avec 8 partenaires."},
        {"Compétence": "Pédagogie & Vulgarisation", "Contexte": "Professeur Particulier", "Réalisation": "Capacité à expliquer des concepts complexes simplement. Hausse des notes de 40%."},
        {"Compétence": "Résilience & Adaptabilité", "Contexte": "Ouvrier Agricole (Nlle-Zélande)", "Réalisation": "Travail en équipe internationale (40 pers) dans un environnement physique exigeant."}
    ]
    
    # Affichage en grille propre
    for skill in skills_data:
        with st.container():
            st.markdown(f"**{skill['Compétence']}**")
            st.caption(f"📍 {skill['Contexte']}")
            st.write(skill['Réalisation'])
            st.divider()

    st.markdown("### 🌍 Langues & Certifications")
    c1, c2, c3 = st.columns(3)
    [cite_start]c1.metric("Anglais", "Courant (C1)", "Cambridge: 186") # [cite: 66]
    [cite_start]c2.metric("Excel", "Expert", "TOSA: 868") # [cite: 67]
    [cite_start]c3.metric("Allemand", "Professionnel", "Notions") # [cite: 9]

# --- TAB 3 : LABO TECHNIQUE (Le code interactif) ---
with tab_tech:
    st.markdown("## ⚡ Pricing d'Option & Structuration")
    st.markdown("""
    En tant que candidat en structuration, je code mes propres outils pour comprendre la mécanique des produits.
    Ci-dessous, mon implémentation du modèle **Black-Scholes** en Python.
    """)
    
    col_input, col_graph = st.columns([1, 2])
    
    with col_input:
        st.markdown('<div class="highlight">Paramètres du Produit</div>', unsafe_allow_html=True)
        current_price = st.number_input("Prix du Sous-jacent (S)", value=100.0, step=1.0)
        strike_price = st.number_input("Strike (K)", value=100.0, step=1.0)
        maturity_days = st.slider("Maturité (Jours)", 1, 365, 30)
        volatility = st.slider("Volatilité Implicite (%)", 5.0, 100.0, 20.0)
        interest_rate = st.number_input("Taux sans risque (%)", value=1.5, step=0.1)
        option_type = st.radio("Type d'Option", ["Call", "Put"], horizontal=True)
        
        # Conversion pour le modèle
        T = maturity_days / 365.0
        r = interest_rate / 100.0
        sigma = volatility / 100.0
        
    # Calculs
    price, delta, gamma, vega, theta, rho = black_scholes(current_price, strike_price, T, r, sigma, option_type)
    
    with col_graph:
        # Affichage des KPIs (Grecques)
        st.markdown("### Valorisation & Sensibilités (Grecques)")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Prix de l'Option", f"{price:.2f} €", delta_color="off")
        kpi2.metric("Delta (Δ)", f"{delta:.3f}", help="Sensibilité au prix du sous-jacent")
        kpi3.metric("Gamma (Γ)", f"{gamma:.4f}", help="Sensibilité du Delta")
        
        kpi4, kpi5, kpi6 = st.columns(3)
        kpi4.metric("Vega (ν)", f"{vega:.3f}", help="Sensibilité à 1% de volatilité")
        kpi5.metric("Theta (Θ)", f"{theta:.3f}", help="Perte de temps par jour")
        kpi6.metric("Rho (ρ)", f"{rho:.3f}", help="Sensibilité aux taux")
        
        st.divider()
        
        # Heatmap
        st.markdown("**🔥 Analyse de Scénarios : Impact Prix (Spot vs Volatilité)**")
        
        # Génération de la matrice pour la Heatmap
        s_range = np.linspace(current_price * 0.85, current_price * 1.15, 10)
        v_range = np.linspace(sigma * 0.5, sigma * 1.5, 10)
        
        heatmap_data = np.zeros((len(v_range), len(s_range)))
        
        for i, v_sim in enumerate(v_range):
            for j, s_sim in enumerate(s_range):
                p_sim, _, _, _, _, _ = black_scholes(s_sim, strike_price, T, r, v_sim, option_type)
                heatmap_data[i, j] = p_sim
                
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(heatmap_data, xticklabels=np.round(s_range, 1), yticklabels=np.round(v_range*100, 1), annot=True, fmt=".1f", cmap="viridis", ax=ax)
        ax.set_xlabel("Spot Price")
        ax.set_ylabel("Volatilité (%)")
        ax.invert_yaxis()
        st.pyplot(fig)