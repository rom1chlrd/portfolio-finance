import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import date
import pydeck as pdk
import yfinance as yf
import plotly.graph_objects as go

# --- CONFIGURATION GÉNÉRALE ---
st.set_page_config(
    page_title="Romain Chalard - Portfolio Structuration",
    page_icon="📈",
    layout="wide"
)

# --- CSS PERSONNALISÉ ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #0E1117; font-weight: 700;}
    .sub-header {font-size: 1.5rem; color: #4F8BF9; font-weight: 600;}
    .highlight {background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 5px solid #4F8BF9;}
    
    section[data-testid="stSidebar"] {
        width: 350px !important; /* On force la largeur à 350px */
    }
</style>
""", unsafe_allow_html=True)

# --- DONNÉES DU CV (Hardcodées pour la simplicité) ---
CONTACT_INFO = {
    "name": "Romain Chalard",
    "tagline": "Étudiant en Ingénierie Financière | Recherche Stage 6 mois - Asset Management",
    "phone": "+33 7 81 78 79 71",
    "email": "romain.chalard@student.junia.com",
    "location": "Paris, France",
    "linkedin": "https://linkedin.com/in/r-chalard", # À modifier
    "github": "https://github.com/rom1chlrd"        # À modifier
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
st.markdown(f'<div class="main-header">Portfolio Technique & Financier | Romain CHALARD</div>', unsafe_allow_html=True)
st.markdown(f"**{CONTACT_INFO['tagline']}**")

# Onglets de navigation
tab_about, tab_market, tab_tech, tab_mc, tab_sales, tab_skills, tab_extra = st.tabs(["👤 À Propos", "📊 Market Data", "💻 Pricer Options", "🎲 Monte Carlo", "📑 Invest. Memo", "💼 Compétences", "🌍 Extra & Perso"])

# --- TAB 1 : À PROPOS & AMBITIONS ---
with tab_about:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Mon Objectif : Investment Management")
        st.info("""
        **Recherche de stage (6 mois) à partir de Juin 2026**
        
        Actuellement en cycle ingénieur à **HEI** et en spécialisation Finance à l'**University of Florida**, 
        je souhaite appliquer mes compétences en Data Analysis et Modélisation à la **gestion d'actifs**.

        Mon approche : Utiliser la donnée (Python/Quant) pour optimiser la prise de décision d'investissement (Fondamentale).
        """)
        
        st.markdown("### Intérêt Personnel pour les Marchés")
        st.write("""
        Au-delà de ma formation académique, je suis un investisseur particulier actif. Cette pratique quotidienne me permet de :
        * **Confronter la théorie à la réalité :** J'applique l'analyse fondamentale (ratios, bilans) et technique pour gérer mon propre portefeuille.
        * **Suivre la Macroéconomie :** Je surveille l'impact des politiques des banques centrales sur les différentes classes d'actifs.
        * **Gérer le Risque :** J'apprends à maîtriser la psychologie de marché et le money management en conditions réelles.
        """)

    with col2:
        st.markdown("### Formation Clé")
        st.markdown("""
        **2026 (Jan-Mai)** 🇺🇸 **University of Florida** *Finance de Marché & Supply Chain*
        
        **2024 - Présent** 🇫🇷 **Junia HEI, Lille** *Ingénierie Financière* *(Maths, VBA, Analyse Financière)*

        **2022 - 2024** 🇫🇷 **Prépa Lycée LaSalle, Lille** *Maths, Physique*
        
        **2019 - 2022** 🇺🇸 **Academica High School** *Dual Diploma (US High School Diploma)*
        """)

# --- TAB 2 : COMPÉTENCES & EXPÉRIENCES ---
with tab_skills:
    st.markdown("### Compétences démontrées")
    
    skills_data = [
        {"Compétence": "Modélisation Mathématique", "Contexte": "Stage Sodexo Bateaux Parisiens", "Réalisation": "Conception d'un modèle complet d'émissions de CO2 sur Excel/VBA pour toute la flotte."},
        {"Compétence": "Prospection & Pitch", "Contexte": "Ambassadeur Agorize", "Réalisation": "Promotion de challenges d'innovation pour des clients Corporate (KPMG, BPCE...). Capacité à convaincre et fédérer."},
        {"Compétence": "Pédagogie & Vulgarisation", "Contexte": "Professeur Particulier", "Réalisation": "Capacité à expliquer des concepts complexes simplement. Hausse des notes de 40% en 3 mois."},
        {"Compétence": "Résilience & Adaptabilité", "Contexte": "Ouvrier Agricole (Nouvelle-Zélande)", "Réalisation": "Travail en équipe internationale (40 pers) dans un environnement physique exigeant."}
    ]
    
    for skill in skills_data:
        st.markdown(f"**{skill['Compétence']}** <span style='color:#666; font-size:0.9em'> — {skill['Contexte']}</span>", unsafe_allow_html=True)
        st.write(skill['Réalisation'])
        
        # ASTUCE : Une ligne de séparation HTML "faite main" avec très peu de marge
        st.markdown("<hr style='margin: 5px 0px 15px 0px; border: none; border-top: 1px solid #e6e6e6;'>", unsafe_allow_html=True)

    # Indicateurs (KPIs)
    st.markdown("### Langues & Certifications")
    c1, c2, c3 = st.columns(3)
    c1.metric("Anglais", "Courant (C1)", "Cambridge: 186")
    c2.metric("Excel", "Avancé", "TOSA: 868")
    c3.metric("Allemand", "Professionnel", "Notions")
    
# --- TAB 3 : PRICER OPTION  ---
with tab_tech:
    st.markdown("## Pricing & Surface de Risque 3D")
    st.write("Visualisation interactive de la sensibilité du prix (Axe Z) par rapport au Spot (Axe X) et à la Volatilité (Axe Y).")
    
    col_input, col_graph = st.columns([1, 3])
    
    with col_input:
        st.markdown('<div class="highlight">Paramètres</div>', unsafe_allow_html=True)
        current_price = st.number_input("Prix Spot (S)", value=100.0, step=1.0)
        strike_price = st.number_input("Strike (K)", value=100.0, step=1.0)
        maturity_days = st.slider("Maturité (Jours)", 1, 365, 30)
        volatility = st.slider("Volatilité (%)", 5.0, 100.0, 20.0)
        interest_rate = st.number_input("Taux (%)", value=1.5, step=0.1)
        option_type = st.radio("Type", ["Call", "Put"], horizontal=True)
        
        T = maturity_days / 365.0
        r = interest_rate / 100.0
        sigma = volatility / 100.0
        
        # Calcul du point actuel
        price, delta, gamma, vega, theta, rho = black_scholes(current_price, strike_price, T, r, sigma, option_type)
        
        st.divider()
        st.metric("Prix Option", f"{price:.2f} €")
        st.metric("Delta (Δ)", f"{delta:.3f}")
        st.metric("Vega (ν)", f"{vega:.3f}")

    with col_graph:
        # --- GÉNÉRATION SURFACE 3D ---
        spot_range = np.linspace(current_price * 0.8, current_price * 1.2, 20)
        vol_range = np.linspace(0.10, 0.60, 20)
        
        X, Y = np.meshgrid(spot_range, vol_range)
        Z = np.zeros_like(X)
        
        for i in range(len(vol_range)):
            for j in range(len(spot_range)):
                p_sim, _, _, _, _, _ = black_scholes(X[i, j], strike_price, T, r, Y[i, j], option_type)
                Z[i, j] = p_sim

        # --- C'EST ICI QUE CA SE PASSE (hovertemplate) ---
        fig = go.Figure(data=[go.Surface(
            z=Z, x=X, y=Y, 
            colorscale='Viridis',
            # Voici la formule magique pour le texte au survol :
            hovertemplate="<b>Spot: %{x:.1f}</b><br>Vol: %{y:.1%}<br>Prix: %{z:.2f} €<extra></extra>"
        )])

        fig.update_layout(
            title=f"Surface de Prix ({option_type})",
            scene=dict(
                xaxis_title='Spot (S)',
                yaxis_title='Volatilité (σ)',
                zaxis_title='Prix (€)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 4 : SIMULATION MONTE CARLO ---
with tab_mc:
    st.markdown("## Simulation Monte-Carlo (Mouvement Brownien)")
    st.markdown("""
    Pour structurer des produits exotiques (ex: Options Asiatiques ou Barrières), les formules fermées ne suffisent plus.
    J'utilise ici `NumPy` pour simuler des milliers de trajectoires possibles du prix de l'actif.
    """)
    
    col_sim_settings, col_sim_graph = st.columns([1, 3])
    
    with col_sim_settings:
        st.markdown('<div class="highlight">Paramètres Simulation</div>', unsafe_allow_html=True)
        n_sims = st.slider("Nombre de scénarios", 10, 1000, 100)
        time_steps = st.slider("Pas de temps (Jours)", 10, 252, 100)
        
        # On reprend les variables globales définies dans l'onglet précédent pour la cohérence
        # Mais on laisse l'utilisateur les ajuster s'il veut tester autre chose ici
        mc_spot = st.number_input("Spot Initial", value=100.0)
        mc_vol = st.slider("Volatilité MC (%)", 5.0, 100.0, 20.0) / 100
        mc_r = st.number_input("Taux sans risque MC (%)", value=1.5) / 100
        mc_T = st.number_input("Horizon (Années)", value=1.0)
        
    with col_sim_graph:
        # LOGIQUE DE CALCUL MONTE CARLO
        # 1. Préparation des variables
        dt = mc_T / time_steps
        S = np.zeros((time_steps + 1, n_sims))
        S[0] = mc_spot
        
        # 2. Génération des chocs aléatoires (Mouvement Brownien)
        # On utilise numpy vectorisé pour la rapidité (pas de boucle for lente)
        Z = np.random.standard_normal((time_steps, n_sims))
        
        # 3. Formule : S(t+1) = S(t) * exp( (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
        drift = (mc_r - 0.5 * mc_vol ** 2) * dt
        diffusion = mc_vol * np.sqrt(dt) * Z
        
        # On calcule les rendements cumulés
        returns = np.exp(drift + diffusion)
        
        # On applique au spot initial (cumprod = produit cumulé)
        S[1:] = mc_spot * np.cumprod(returns, axis=0)
        
        # VISUALISATION
        fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
        ax_mc.plot(S[:, :100], alpha=0.4, linewidth=1) # On affiche max 100 lignes pour pas surcharger
        ax_mc.set_title(f"Projection de {n_sims} scénarios sur {mc_T} an(s)")
        ax_mc.set_xlabel("Jours de trading")
        ax_mc.set_ylabel("Prix de l'actif")
        ax_mc.grid(True, alpha=0.3)
        
        # Afficher la moyenne (Espérance)
        mean_path = np.mean(S, axis=1)
        ax_mc.plot(mean_path, color='black', linewidth=2, linestyle='--', label="Moyenne")
        ax_mc.legend()
        
        st.pyplot(fig_mc)
        
        # KPI Finale
        final_mean = mean_path[-1]
        st.metric("Prix moyen à maturité", f"{final_mean:.2f} €", delta=f"{((final_mean/mc_spot)-1)*100:.2f}% vs Spot")

# --- TAB MARKET : ASSET MANAGEMENT & OPTIMISATION ---
with tab_market:
    st.markdown("## Construction de Portefeuille (Markowitz)")
    st.markdown("""
    En Asset Management, l'objectif est de maximiser le rendement pour un niveau de risque donné.
    J'utilise ici la **Frontière Efficiente** pour visualiser l'allocation optimale d'un panier d'actifs.
    """)
    
    col_sel, col_viz = st.columns([1, 3])
    
    with col_sel:
        st.markdown('<div class="highlight">Sélection Assets</div>', unsafe_allow_html=True)
        default_tickers = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM']
        tickers = st.multiselect("Univers d'investissement", default_tickers, default=default_tickers[:4])
        period = st.selectbox("Historique", ["1y", "2y", "5y"], index=0)
        st.caption("Données Live Yahoo Finance")
        
    with col_viz:
        if len(tickers) > 1:
            try:
                # 1. Récupération des Data
                data = yf.download(tickers, period=period)['Close']
                returns = data.pct_change().dropna()
                
                mean_returns = returns.mean() * 252
                cov_matrix = returns.cov() * 252
                
                # 2. Simulation de 2000 Portefeuilles (Monte Carlo)
                num_portfolios = 2000
                results = np.zeros((3, num_portfolios))
                
                for i in range(num_portfolios):
                    weights = np.random.random(len(tickers))
                    weights /= np.sum(weights)
                    
                    p_return = np.sum(weights * mean_returns)
                    p_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    
                    results[0,i] = p_return
                    results[1,i] = p_std_dev
                    results[2,i] = results[0,i] / results[1,i] # Sharpe Ratio (Simplified, Rf=0)

                # 3. Visualisation Plotly
                max_sharpe_idx = np.argmax(results[2])
                sdp, rp = results[1,max_sharpe_idx], results[0,max_sharpe_idx]
                
                fig = go.Figure()
                
                # Nuage de points
                fig.add_trace(go.Scatter(
                    x=results[1,:], y=results[0,:], mode='markers',
                    marker=dict(color=results[2,:], colorscale='Viridis', showscale=True, size=5),
                    name='Portefeuilles simulés'
                ))
                
                # Point optimal
                fig.add_trace(go.Scatter(
                    x=[sdp], y=[rp], mode='markers',
                    marker=dict(color='red', size=15, symbol='star'),
                    name='Max Sharpe Ratio'
                ))
                
                fig.update_layout(
                    title="Frontière Efficiente (Risk vs Return)",
                    xaxis_title="Volatilité (Risque Annuel)",
                    yaxis_title="Rendement Espéré (Annuel)",
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(r=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f" **Portefeuille Optimal :** Rendement {rp:.1%} | Volatilité {sdp:.1%}")

            except Exception as e:
                st.error(f"Erreur data : {e}")
        else:
            st.warning("Sélectionnez au moins 2 actifs.")



# --- TAB 6 : MARKET DATA & CORRELATION --- (pas sur le site pour l'instant)
if False:
    
    with tab_market:
        st.markdown("## Analyse de Marché (Données Réelles)")
        st.markdown("""
        En structuration, on travaille souvent sur des paniers d'actifs (Basket Options). 
        Comprendre la corrélation entre les sous-jacents est crucial pour pricer le risque.
        
        *Les données ci-dessous sont récupérées en temps réel via l'API Yahoo Finance.*
        """)
        
        col_sel, col_viz = st.columns([1, 3])
        
        with col_sel:
            st.markdown('<div class="highlight">Sélection du Panier</div>', unsafe_allow_html=True)
            # Liste de tickers par défaut (CAC40 & Tech US)
            default_tickers = ['AC.PA', 'MC.PA', 'TEP.PA', 'AAPL', 'MSFT', 'NVDA']
            tickers = st.multiselect("Choix des Actions", default_tickers, default=default_tickers[:4])
            period = st.selectbox("Période d'analyse", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
            
        with col_viz:
            if len(tickers) > 1:
                try:
                    # Téléchargement des données
                    data = yf.download(tickers, period=period)['Close']
                    
                    # Calcul des rendements quotidiens (Log returns)
                    returns = np.log(data / data.shift(1)).dropna()
                
                    # Calcul de la corrélation
                    corr_matrix = returns.corr()
                    
                    # Affichage 1 : La Heatmap
                    st.subheader("Matrice de Corrélation")
                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax_corr)
                    st.pyplot(fig_corr)
                    
                    st.divider()
                    
                    # Affichage 2 : Performance comparée
                    st.subheader("Performance Relative (Base 100)")
                    # Normalisation base 100 pour comparer
                    normalized_data = (data / data.iloc[0]) * 100
                    st.line_chart(normalized_data)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la récupération des données. Vérifiez les tickers. ({e})")
            else:
                st.warning("Veuillez sélectionner au moins 2 actifs pour afficher la corrélation.")

# --- TAB MEMO : INVESTMENT THESIS ---
with tab_sales: 
    st.markdown("## Générateur de Thèse d'Investissement")
    st.write("Simulation d'un mémo de recherche pour le Comité d'Investissement.")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        ticker_memo = st.text_input("Ticker (ex: NVDA)", "NVDA")
        reco = st.selectbox("Recommandation", ["BUY", "HOLD", "SELL"])
        horizon = st.selectbox("Horizon", ["Court terme (Tactique)", "Long Terme (Stratégique)"])
        catalyst = st.text_area("Catalyseur Principal", "Avance technologique sur l'IA et demande forte des Data Centers.")
        risk = st.text_area("Risque Principal", "Valorisation tendue et concurrence accrue.")
        
    with c2:
        st.markdown("###  Aperçu du Mémo")
        
        memo_html = f"""
<div style="background-color: white; padding: 25px; border: 1px solid #d1d5db; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: #1f2937; font-family: sans-serif;">
    <div style="border-bottom: 2px solid #2563eb; padding-bottom: 10px; margin-bottom: 20px;">
        <h3 style="margin: 0; color: #1e3a8a; text-align: center; text-transform: uppercase;">Investment Committee Memo</h3>
    </div>
    <div style="background-color: #f3f4f6; padding: 15px; border-radius: 5px; margin-bottom: 20px; font-size: 0.9em;">
        <p style="margin: 5px 0;"><strong> DATE :</strong> {date.today().strftime('%d/%m/%Y')}</p>
        <p style="margin: 5px 0;"><strong> FROM :</strong> Romain Chalard (Analyste Junior)</p>
        <p style="margin: 5px 0;"><strong> ASSET :</strong> <span style="color: #2563eb; font-weight: bold;">{ticker_memo}</span></p>
        <p style="margin: 5px 0;"><strong> ACTION :</strong> <span style="background-color: {'#dcfce7' if reco == 'BUY' else '#fee2e2'}; color: {'#166534' if reco == 'BUY' else '#991b1b'}; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{reco}</span> ({horizon})</p>
    </div>
    <h4 style="color: #374151; border-left: 4px solid #2563eb; padding-left: 10px; margin-top: 20px;">1. Thèse d'Investissement</h4>
    <p style="line-height: 1.6; color: #4b5563;">{catalyst}</p>
    <h4 style="color: #374151; border-left: 4px solid #ef4444; padding-left: 10px; margin-top: 20px;">2. Facteurs de Risque</h4>
    <p style="line-height: 1.6; color: #4b5563;">{risk}</p>
    <div style="margin-top: 25px; padding: 15px; background-color: #eff6ff; border-left: 4px solid #2563eb;">
        <strong> Conclusion :</strong><br>
        Compte tenu de l'analyse fondamentale et quantitative, nous recommandons d'initier une position <strong>{reco}</strong> sur ce titre.
    </div>
</div>
"""
        st.markdown(memo_html, unsafe_allow_html=True)
        
# --- TAB SALES --- (pas sur le site)
if False:
    with tab_sales:
        st.markdown("## Cockpit Sales & Structuration")
        st.markdown("""
        Ici, je simule l'approche d'un Sales CIB : **Comprendre le besoin client - Structurer une solution - Pricer - Pitcher.**
        """)
        
        # Division en 2 colonnes : Paramètres Client (Gauche) / Solution & Visuel (Droite)
        col_input, col_output = st.columns([1, 2])
        
        with col_input:
            st.markdown('<div class="highlight">1. Paramètres du Produit</div>', unsafe_allow_html=True)
            
            # Choix de base
            underlying = st.selectbox("Sous-jacent", ["Euro Stoxx 50", "S&P 500", "LVMH", "TotalEnergies", "Nvidia"])
            product_type = st.selectbox("Structure", ["Phoenix Mémoire (Yield)", "Autocall Athena (Early Redemp.)", "Capital Garanti (Call Spread)"])
            
            st.markdown("---")
            st.markdown("**Paramètres de Structuration**")
            
            # Paramètres dynamiques selon le produit
            maturity = st.slider("Maturité (Années)", 1, 10, 5)
            
            if "Phoenix" in product_type or "Autocall" in product_type:
                barrier_protection = st.slider("Barrière de Protection (Capital)", 40, 80, 60, help="Niveau en % du prix initial en dessous duquel le capital est à risque")
                barrier_coupon = st.slider("Barrière de Coupon", 50, 100, 70, help="Niveau pour toucher le coupon")
                autocall_trigger = st.number_input("Niveau d'Autocall (%)", value=100)
            else: # Capital Garanti
                participation = st.slider("Participation à la hausse (%)", 50, 150, 100)
                protection = 100 # Capital garanti
                
            st.markdown("---")
            market_env = st.selectbox("Environnement de Volatilité", ["Faible (<15%)", "Moyenne (15-25%)", "Élevée (>25%)"])
    
        with col_output:
            st.markdown('<div class="highlight">2. Structuration & Pitch</div>', unsafe_allow_html=True)
            
            # --- MOTEUR DE PRICING SIMULÉ (Logique Heuristique pour la démo) ---
            # Note pour le recruteur : Ceci est une simulation de logique de pricing pour démontrer la mécanique
            base_coupon = 5.0 # Taux sans risque approx + spread
            
            # Impact Volatilité
            vol_impact = 0
            if market_env == "Élevée (>25%)": vol_impact = 3.0
            elif market_env == "Moyenne (15-25%)": vol_impact = 1.5
            
            # Impact Structure
            struct_yield = 0
            if "Phoenix" in product_type:
                # Plus la barrière est haute (risque), plus le coupon est bas. Plus la barrière est basse (safe), plus le coupon est bas... wait no.
                # En vente d'option (Phoenix) : Plus on prend de risque (Barrière haute), plus on a de rendement ? Non, c'est l'inverse en structuration.
                # Plus la barrière est basse (ex: 50%), plus l'option de vente vaut cher ? Non.
                # Simplification : Plus la protection est "loin" (40%), plus le coupon est FAIBLE (car moins risqué).
                # Plus la protection est proche (80%), plus le coupon est ÉLEVÉ (car risqué).
                risk_premium = (barrier_protection - 40) * 0.15 
                struct_yield = base_coupon + vol_impact + risk_premium
                display_metric = f"{struct_yield:.2f}% / an"
                metric_label = "Coupon Indicatif"
                
            elif "Garanti" in product_type:
                # Capital Garanti : On paie pour la garantie, donc rendement faible ou participation ajustée
                struct_yield = (participation / 100) * (base_coupon + vol_impact) 
                # C'est un peu faux mathématiquement mais logique commercialement pour la démo
                display_metric = f"{participation}%"
                metric_label = "Participation à la hausse"
                
            else: # Athena
                struct_yield = base_coupon + vol_impact + 2.0 # Prime d'autocall
                display_metric = f"{struct_yield:.2f}% / an"
                metric_label = "Rendement si Rappel"

            # --- VISUALISATION (Term Sheet & Graph) ---
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.metric(label=metric_label, value=display_metric, delta="Pricing Live")
                st.info(f"**Protection :** Jusqu'à -{100-barrier_protection}%" if "Garanti" not in product_type else "**Capital Garanti 100%**")
            
            with c2:
                # Graphique de Payoff à Maturité
                fig_payoff, ax_p = plt.subplots(figsize=(6, 3))
                spots = np.linspace(0, 150, 100)
                payoffs = np.zeros_like(spots)
                
                if "Phoenix" in product_type:
                    # Si Spot > Barrière Capital : 100% + Coupon (simplifié)
                    # Si Spot < Barrière Capital : Perte en capital
                    barrier_val = barrier_protection
                    for i, s in enumerate(spots):
                        if s >= barrier_val:
                            payoffs[i] = 100 + struct_yield # On récupère 100 + le coupon
                        else:
                            payoffs[i] = s # On perd (remboursement à la valeur de l'action)
                    
                    ax_p.plot(spots, payoffs, color='#4F8BF9', linewidth=2, label='Remboursement')
                    ax_p.axvline(barrier_val, color='red', linestyle='--', label=f'Barrière (-{100-barrier_val}%)')
                    ax_p.axhline(100, color='gray', linestyle=':', linewidth=0.5)
                    ax_p.fill_between(spots, 0, payoffs, alpha=0.1, color='#4F8BF9')
    
                elif "Garanti" in product_type:
                    # Call Spread : Max(100, 100 + Participation * (S-100))
                    payoffs = [100 + max(0, participation/100 * (s - 100)) for s in spots]
                    ax_p.plot(spots, payoffs, color='green', linewidth=2)
                    ax_p.axhline(100, color='green', linestyle='--', label='Capital Garanti')
    
                ax_p.set_title("Scénario à Maturité (Payoff)", fontsize=10)
                ax_p.set_xlabel("% du Prix Initial")
                ax_p.set_ylabel("Remboursement (%)")
                ax_p.legend(fontsize=8)
                ax_p.grid(True, alpha=0.3)
                st.pyplot(fig_payoff)
    
            # --- GENERATEUR DE PITCH ---
            st.markdown("### Email Client (Généré)")
            pitch_text = f"""
            Objet : Opportunité {product_type} sur {underlying} - Coupon {display_metric}
            
            Bonjour,
            
            Dans le contexte actuel de volatilité {market_env.split('(')[0].lower()}, nous avons structuré une solution pour optimiser le rendement de votre poche actions.
            
            La Proposition : {underlying}
            1. Rendement : {metric_label} cible de {display_metric}.
            2. Protection : Le capital est protégé jusqu'à une baisse de {100-barrier_protection if "Garanti" not in product_type else 0}% à maturité.
            3. Mécanisme : { "Coupons mémorisables versés si l'action tient la barrière de " + str(barrier_coupon) + "%." if "Phoenix" in product_type else "Participation à la hausse avec 0 risque en capital."}
            
            C'est le moment idéal pour pricer cette structure car la volatilité nous permet d'aller chercher ce niveau de coupon attractif.
            Je suis disponible pour en discuter de vive voix et ajuster les paramètres selon vos contraintes.
            
            Bien à vous,
            Romain Chalard
            """
            st.text_area("Draft prêt à envoyer :", value=pitch_text.replace("        ", ""), height=250)

# --- TAB : EXTRA & PERSO ---
with tab_extra:
    st.markdown("## Profil International & Leadership")
    st.write("Mon parcours est marqué par une forte mobilité internationale et des responsabilités associatives.")

    col_map, col_lifestyle = st.columns([2, 1])

    with col_map:
        st.markdown("### Carte de mes expériences")
        
        # 1. Vos données (Mêmes coordonnées qu'avant)
        map_data = pd.DataFrame({
            'lat': [50.629, -37.783, 29.651, 25.761],
            'lon': [3.057, 176.316, -82.324, -80.191],
            'Lieu': ['Lille (Junia HEI)', 'Te Puke (Kiwi Harvest)', 'Gainesville (UF Exchange)', 'Miami (High School Diploma)']
        })

        # 2. Configuration de la carte "Custom"
        # On crée une couche de points (Scatterplot)
        layer = pdk.Layer(
            "ScatterplotLayer",
            map_data,
            get_position='[lon, lat]',
            get_color=[255, 75, 75, 200],  # Couleur Rouge [R, G, B, Transparence]
            get_radius=300000,             # Rayon des points en mètres (300km pour être gros sur la carte monde)
            pickable=True                  # Permet d'afficher le texte au survol
        )

        # 3. Vue initiale (Zoom dézoomé pour voir le monde)
        view_state = pdk.ViewState(
            latitude=10,
            longitude=0,
            zoom=0.3,
            pitch=0,
        )

        # 4. Affichage de la carte avec une infobulle (Tooltip)
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=view_state,
            layers=[layer],
            tooltip={"text": "{Lieu}"} # Affiche le nom quand on passe la souris dessus !
        ))
        
        st.caption("""
        📍 **Lille** : Cycle Ingénieur (Actuel)
        📍 **Miami** : Dual Diploma High School (2019-2022)
        📍 **Gainesville (Floride)** : Semestre d'échange à l'University of Florida (Jan 2026)
        📍 **Te Puke (NZ)** : Ouvrier agricole saisonnier (2025)
        """)

    with col_lifestyle:
        st.markdown("### Leadership")
        st.info("**Président Club Oenologie**")
        st.write("""
        Une aventure humaine et entrepreneuriale :
        * **Management :** Recrutement et coordination de 20 membres actifs.
        * **Event :** Organisation de dégustations pour 400 étudiants.
        * **Gestion :** Pilotage d'un budget de 6 000 €/an et partenariats.
        """)
        
        st.divider()
        
        st.markdown("### Compétition")
        st.write("""
        **Ski de Compétition :** Cette discipline m'a appris la résilience et la prise de risque calculée, des qualités que je transpose aujourd'hui dans la finance de marché.
        """)
