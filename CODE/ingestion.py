"""
Module 1: Data Ingestion
Fetches and structures patent data from PatentsView API or generates a synthetic corpus.
"""

import os
import json
import random
import requests
import pandas as pd

# -- Synthetic corpus (fallback when network is unavailable) -----------------
SYNTHETIC_PATENTS = [
    # Machine Learning / AI
    ("US10001", "Neural network training with sparse gradients",
     "A method for training deep neural networks using sparse gradient updates to reduce computation. The system applies adaptive learning rates and momentum to accelerate convergence in classification tasks."),
    ("US10002", "Convolutional architecture for image recognition",
     "An improved convolutional neural network architecture for image classification incorporating residual connections, batch normalization, and depthwise separable convolutions to achieve state-of-the-art accuracy."),
    ("US10003", "Transformer model for natural language processing",
     "A transformer-based architecture employing self-attention mechanisms for natural language understanding. The model pre-trains on large text corpora and fine-tunes on downstream tasks including question answering and sentiment analysis."),
    ("US10004", "Reinforcement learning for robotic control",
     "A reinforcement learning framework for training robotic agents using policy gradient methods. The system incorporates reward shaping and hierarchical action spaces to improve sample efficiency in manipulation tasks."),
    ("US10005", "Federated learning for privacy-preserving AI",
     "A federated learning approach enabling model training across distributed devices without sharing raw data. The system uses differential privacy and secure aggregation protocols to protect user information."),
    ("US10006", "Graph neural networks for molecular property prediction",
     "A graph neural network framework for predicting molecular properties from chemical structures. The system encodes atoms as nodes and bonds as edges, learning representations for drug discovery applications."),
    ("US10007", "Meta-learning for few-shot classification",
     "A meta-learning algorithm enabling classifiers to generalize from few labeled examples. The system learns task-agnostic representations that rapidly adapt to novel categories through gradient-based optimization."),
    ("US10008", "Generative adversarial network for synthetic data",
     "A generative adversarial network architecture producing high-fidelity synthetic datasets for data augmentation. The discriminator employs spectral normalization to stabilize training and improve output quality."),

    # Semiconductor / Electronics
    ("US20001", "Low-power CMOS transistor design for mobile chips",
     "A complementary metal-oxide-semiconductor transistor design with reduced leakage current for mobile applications. The device employs high-k dielectric materials and multi-gate architectures to improve power efficiency."),
    ("US20002", "Quantum dot display with improved color gamut",
     "A quantum dot display technology achieving wide color gamut through cadmium-free nanocrystals. The system uses photoluminescence tuning to produce precise emission wavelengths for red, green, and blue subpixels."),
    ("US20003", "3D NAND flash memory with vertical channel",
     "A three-dimensional NAND flash memory architecture with vertically stacked cells increasing storage density. The fabrication process uses atomic layer deposition to form uniform charge-trap layers across all tiers."),
    ("US20004", "High-electron-mobility transistor for RF applications",
     "A gallium nitride high-electron-mobility transistor optimized for radio-frequency power amplification. The device features a recessed gate structure and passivation layer to minimize current collapse."),
    ("US20005", "Silicon photonics integrated circuit for data centers",
     "A silicon photonics platform integrating optical waveguides, modulators, and photodetectors on a single chip for high-bandwidth data center interconnects. The system achieves terabit-per-second throughput."),
    ("US20006", "Memristor-based neuromorphic computing circuit",
     "A memristor crossbar array implementing synaptic weights for neuromorphic computing. The circuit performs matrix-vector multiplication in analog domain, reducing energy consumption for inference tasks."),
    ("US20007", "Flexible electronics substrate for wearable sensors",
     "A flexible electronic substrate based on polyimide film with embedded silver nanowire conductors enabling stretchable sensor arrays. The device maintains electrical performance under repeated mechanical deformation."),
    ("US20008", "Phase-change memory with improved retention",
     "A phase-change memory device using germanium-antimony-telluride alloy achieving improved data retention at elevated temperatures. The cell design incorporates a confined geometry to reduce reset current requirements."),

    # Biotechnology
    ("US30001", "CRISPR-Cas9 gene editing with enhanced specificity",
     "A CRISPR-Cas9 gene editing system with engineered nuclease variants exhibiting enhanced on-target specificity. The method employs truncated guide RNAs and high-fidelity Cas9 mutants to minimize off-target cleavage."),
    ("US30002", "mRNA vaccine delivery using lipid nanoparticles",
     "An mRNA vaccine platform utilizing ionizable lipid nanoparticles for efficient intracellular delivery. The formulation achieves high encapsulation efficiency and controlled release to elicit robust immune responses."),
    ("US30003", "Antibody-drug conjugate for targeted cancer therapy",
     "An antibody-drug conjugate linking a tumor-targeting antibody to a cytotoxic payload via a cleavable linker. The construct selectively delivers chemotherapy to antigen-expressing cancer cells while sparing healthy tissue."),
    ("US30004", "Microfluidic device for single-cell RNA sequencing",
     "A microfluidic droplet device enabling high-throughput single-cell RNA sequencing. The system encapsulates individual cells with barcoded hydrogel beads for parallel library preparation and transcriptome profiling."),
    ("US30005", "Protein engineering via directed evolution",
     "A directed evolution platform for engineering proteins with enhanced thermostability and catalytic activity. The method combines error-prone PCR mutagenesis with high-throughput fluorescence screening to identify improved variants."),
    ("US30006", "Biosensor for real-time glucose monitoring",
     "An electrochemical biosensor for continuous glucose monitoring based on glucose oxidase immobilized on a graphene electrode. The device achieves high sensitivity and selectivity in interstitial fluid measurements."),
    ("US30007", "CAR-T cell therapy for hematologic malignancies",
     "A chimeric antigen receptor T-cell therapy targeting CD19-expressing leukemia cells. The CAR construct incorporates co-stimulatory domains to enhance T-cell persistence and anti-tumor efficacy."),
    ("US30008", "Organoid model for drug toxicity screening",
     "A liver organoid model derived from human induced pluripotent stem cells for high-throughput drug toxicity screening. The system recapitulates key hepatic functions including albumin secretion and cytochrome P450 metabolism."),

    # Renewable Energy
    ("US40001", "Perovskite solar cell with improved stability",
     "A perovskite solar cell incorporating cesium-formamidinium lead halide absorber layers achieving improved thermal and moisture stability. The device uses a hole transport layer modification to reduce ion migration."),
    ("US40002", "Solid-state lithium battery with ceramic electrolyte",
     "A solid-state lithium battery employing a sulfide-based ceramic electrolyte achieving high ionic conductivity at room temperature. The cell architecture addresses interfacial resistance through in-situ coating techniques."),
    ("US40003", "Wind turbine blade with adaptive geometry",
     "A wind turbine blade with adaptive trailing edge geometry for load control across variable wind conditions. The system uses piezoelectric actuators to dynamically adjust blade camber and reduce fatigue loads."),
    ("US40004", "Proton exchange membrane fuel cell optimization",
     "An optimized proton exchange membrane fuel cell with platinum-group metal-free cathode catalysts. The system employs nitrogen-doped carbon supports to enhance oxygen reduction reaction kinetics and durability."),
    ("US40005", "Thermophotovoltaic system for waste heat recovery",
     "A thermophotovoltaic system converting industrial waste heat to electricity using tungsten photonic crystal emitters and gallium antimonide photovoltaic cells. The system achieves high spectral efficiency."),
    ("US40006", "Hydrogen production via photoelectrochemical water splitting",
     "A photoelectrochemical cell for solar hydrogen production using bismuth vanadate photoanodes. The system incorporates a cobalt phosphate oxygen evolution catalyst to improve charge transfer efficiency."),
    ("US40007", "Grid-scale vanadium redox flow battery",
     "A vanadium redox flow battery system for grid-scale energy storage featuring an improved ion exchange membrane. The system achieves high cycle stability and rapid response for frequency regulation applications."),
    ("US40008", "Concentrated solar power with molten salt storage",
     "A concentrated solar power plant using parabolic trough collectors and molten salt thermal energy storage. The system enables dispatchable electricity generation during periods of low solar irradiance."),

    # Autonomous Vehicles
    ("US50001", "LiDAR sensor fusion for autonomous navigation",
     "A sensor fusion architecture combining LiDAR point clouds with camera imagery for autonomous vehicle navigation. The system employs deep learning-based object detection and tracking for real-time scene understanding."),
    ("US50002", "Path planning algorithm for urban driving",
     "A motion planning algorithm for autonomous vehicles in complex urban environments using probabilistic roadmaps. The system incorporates pedestrian behavior prediction and traffic rule compliance for safe navigation."),
    ("US50003", "Vehicle-to-everything communication protocol",
     "A vehicle-to-everything communication protocol enabling low-latency data exchange between autonomous vehicles and infrastructure. The system uses dedicated short-range communication to coordinate intersection management."),
    ("US50004", "Simultaneous localization and mapping for robotics",
     "A simultaneous localization and mapping system for autonomous mobile robots using graph-based optimization. The algorithm fuses wheel odometry with visual and inertial measurements for accurate pose estimation."),
    ("US50005", "Electric vehicle battery management system",
     "A battery management system for electric vehicles monitoring cell-level voltage, current, and temperature. The system employs adaptive state-of-charge estimation using extended Kalman filtering for accurate range prediction."),
    ("US50006", "Radar-based collision avoidance system",
     "A frequency-modulated continuous-wave radar system for autonomous vehicle collision avoidance. The system detects and classifies objects at long range while maintaining performance in adverse weather conditions."),
    ("US50007", "High-definition mapping for self-driving vehicles",
     "A high-definition mapping system for autonomous vehicles aggregating crowd-sourced sensor data. The platform maintains centimeter-accurate lane geometry and semantic annotations for navigation."),
    ("US50008", "Deep learning perception for pedestrian detection",
     "A deep learning model for pedestrian detection in autonomous driving scenarios using multi-scale feature pyramids. The system achieves high recall under occlusion and varying illumination conditions."),

    # Medical Devices
    ("US60001", "Implantable neural interface for brain-computer interface",
     "An implantable neural interface with high-density electrode arrays for brain-computer interface applications. The device uses wireless telemetry to transmit neural signals with high bandwidth and low power consumption."),
    ("US60002", "Robotic surgical system with haptic feedback",
     "A robotic surgical system providing haptic force feedback to the surgeon during minimally invasive procedures. The system integrates force sensors at instrument tips with haptic actuators at master controls."),
    ("US60003", "Wearable cardiac monitor with arrhythmia detection",
     "A wearable cardiac monitor using deep learning algorithms for real-time arrhythmia detection from single-lead ECG. The device transmits alerts to clinicians and stores continuous recordings for retrospective analysis."),
    ("US60004", "Optical coherence tomography for ophthalmology",
     "An optical coherence tomography system for high-resolution retinal imaging incorporating swept-source laser technology. The device achieves micrometer-resolution cross-sectional imaging for glaucoma and macular degeneration diagnosis."),
    ("US60005", "Drug-eluting stent with controlled release coating",
     "A drug-eluting coronary stent with a biodegradable polymer coating enabling controlled release of antiproliferative agents. The device reduces restenosis rates while promoting endothelialization after implantation."),
    ("US60006", "3D-printed patient-specific orthopedic implant",
     "A titanium orthopedic implant fabricated by selective laser melting with patient-specific geometry derived from CT imaging. The porous scaffold promotes osseointegration through biomimetic trabecular architecture."),
    ("US60007", "Continuous glucose monitoring with closed-loop insulin delivery",
     "A closed-loop insulin delivery system combining continuous glucose monitoring with an algorithm-controlled insulin pump. The system uses model predictive control to maintain euglycemia in type 1 diabetes patients."),
    ("US60008", "Ultrasound-guided drug delivery using microbubbles",
     "An ultrasound-triggered drug delivery system using microbubble carriers for targeted therapeutic delivery. Acoustic cavitation enhances membrane permeability to facilitate intracellular drug uptake at the target site."),
]

CPC_MAP = {
    "US1": "G06N",   # AI/ML
    "US2": "H01L",   # Semiconductor
    "US3": "C12N",   # Biotech
    "US4": "H02S",   # Renewable Energy
    "US5": "B60W",   # Autonomous Vehicles
    "US6": "A61B",   # Medical Devices
}


def _cpc(patent_id: str) -> str:
    for prefix, code in CPC_MAP.items():
        if patent_id.startswith(prefix):
            return code
    return "UNKNOWN"


def load_synthetic_corpus(n: int = 48) -> pd.DataFrame:
    """Return a DataFrame of synthetic patents (up to len(SYNTHETIC_PATENTS))."""
    pool = SYNTHETIC_PATENTS[:n] if n <= len(SYNTHETIC_PATENTS) else SYNTHETIC_PATENTS
    records = []
    for pid, title, abstract in pool:
        records.append({
            "patent_id": pid,
            "title": title,
            "abstract": abstract,
            "cpc_code": _cpc(pid),
            "text": title + ". " + abstract,
        })
    df = pd.DataFrame(records)
    print(f"[Ingestion] Loaded {len(df)} synthetic patents.")
    return df


def fetch_patentsview(n: int = 200, per_page: int = 100) -> pd.DataFrame:
    """
    Attempt to fetch real patents from the PatentsView API.
    Falls back to the synthetic corpus on any error.
    """
    base = "https://api.patentsview.org/patents/query"
    fields = ["patent_id", "patent_title", "patent_abstract"]
    records = []
    page = 1
    fetched = 0
    print("[Ingestion] Attempting PatentsView API fetch ...")
    while fetched < n:
        size = min(per_page, n - fetched)
        payload = {
            "q": {"_gte": {"patent_date": "2018-01-01"}},
            "f": fields,
            "o": {"page": page, "per_page": size},
        }
        try:
            r = requests.post(base, json=payload, timeout=20)
            r.raise_for_status()
            patents = r.json().get("patents") or []
            if not patents:
                break
            for p in patents:
                ab = (p.get("patent_abstract") or "").strip()
                ti = (p.get("patent_title") or "").strip()
                if ab and ti:
                    records.append({
                        "patent_id": p.get("patent_id", ""),
                        "title": ti,
                        "abstract": ab,
                        "cpc_code": "UNKNOWN",
                        "text": ti + ". " + ab,
                    })
            fetched += len(patents)
            page += 1
            if len(patents) < size:
                break
        except Exception as exc:
            print(f"[Ingestion] API error: {exc}. Falling back to synthetic corpus.")
            return load_synthetic_corpus()

    if not records:
        print("[Ingestion] No records returned. Falling back to synthetic corpus.")
        return load_synthetic_corpus()

    df = pd.DataFrame(records)
    print(f"[Ingestion] Fetched {len(df)} patents from PatentsView.")
    return df


def load_bigpatent(n: int = 500) -> pd.DataFrame:
    """
    Load real USPTO patents from the HuggingFace big_patent dataset.
    Pulls n//5 patents from each of 5 CPC sections covering the 6 project domains.
    Falls back to synthetic corpus if the dataset is unavailable.
    """
    section_map = [
        ("g", "G06N", "Machine Learning / AI"),
        ("h", "H01L", "Semiconductor / Electronics"),
        ("a", "A61B", "Biotechnology / Medical"),
        ("b", "B60W", "Autonomous Vehicles"),
        ("f", "H02S", "Renewable Energy"),
    ]
    per_section = max(1, n // len(section_map))
    records = []

    try:
        from datasets import load_dataset
    except ImportError:
        print("[Ingestion] 'datasets' package not installed. Run: pip install datasets")
        return load_synthetic_corpus(n)

    for sec, cpc, domain in section_map:
        print(f"[Ingestion] Loading section '{sec}' ({domain}) ...")
        try:
            ds = load_dataset("big_patent", sec, split="train", streaming=True)
            count = 0
            for ex in ds:
                abstract = (ex.get("abstract") or "").strip()
                if len(abstract) < 40:
                    continue
                # Derive a title from the first sentence of the abstract
                first_sent = abstract.split(". ")[0]
                title = first_sent[:100].strip().rstrip(".")
                if len(title) < 10:
                    title = abstract[:80].strip()
                pid = f"{sec.upper()}{cpc[:3]}{len(records)+1:05d}"
                records.append({
                    "patent_id": pid,
                    "title":     title,
                    "abstract":  abstract,
                    "cpc_code":  cpc,
                    "text":      title + ". " + abstract,
                })
                count += 1
                if count >= per_section:
                    break
            print(f"  Loaded {count} patents from section '{sec}'.")
        except Exception as exc:
            print(f"[Ingestion] Section '{sec}' failed: {exc}. Skipping.")

    if not records:
        print("[Ingestion] big_patent unavailable. Falling back to synthetic corpus.")
        return load_synthetic_corpus(n)

    df = pd.DataFrame(records)
    print(f"[Ingestion] Loaded {len(df)} real patents from big_patent (HuggingFace).")
    return df


def load_corpus(source: str = "synthetic", n: int = 48) -> pd.DataFrame:
    """
    Main entry point.
    source: 'synthetic' | 'patentsview' | 'bigpatent'
    """
    if source == "patentsview":
        return fetch_patentsview(n=n)
    if source == "bigpatent":
        return load_bigpatent(n=n)
    return load_synthetic_corpus(n=n)


if __name__ == "__main__":
    df = load_corpus("synthetic")
    print(df[["patent_id", "cpc_code", "title"]].to_string(index=False))
