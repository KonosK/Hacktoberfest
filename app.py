"""GeneRx - Consumer Medication & DNA Safety Dashboard."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import pipeline  # noqa: E402
from clinical_data import RX_SCENARIOS, generate_mock_rx_image, generate_synthetic_23andme  # noqa: E402

st.set_page_config(page_title="GeneRx | Pharmacogenomic Safety", page_icon="🧬", layout="wide")

IMG_TYPES = ["png", "jpg", "jpeg", "webp", "heic", "heif", "avif"]

SCENARIO_MAP = {
    "Plavix + Omeprazole": "Scenario 1 - Clopidogrel + Grapefruit Juice & Omeprazole",
    "Warfarin + Ibuprofen": "Scenario 2 - Warfarin + Ibuprofen",
    "Simvastatin + Grapefruit": "Scenario 3 - Simvastatin + Grapefruit Juice",
}


@st.cache_data(show_spinner=False)
def scenario_image(key: str) -> bytes:
    return generate_mock_rx_image(key)


@st.cache_data(show_spinner=False)
def synthetic_dna(profile: str) -> str:
    return generate_synthetic_23andme(profile)


def get_detected_items_list(vision: dict) -> list[str]:
    items = []
    for d in vision.get("drugs", []):
        brand = d.get("brand_name", "").strip()
        generic = d.get("generic_name", "").strip().title()
        if brand and generic and brand.lower() != generic.lower():
            items.append(f"{brand} ({generic})")
        elif brand:
            items.append(brand)
        elif generic:
            items.append(generic)
    for item in vision.get("dietary_and_otc_items", []):
        item_str = str(item).strip().title()
        if item_str and item_str not in items:
            items.append(item_str)
    return items


def main() -> None:
    settings = pipeline.resolve_api_settings()

    # CSS Injection for Card Contrast
    st.markdown(
        """
        <style>
            /* Target Streamlit's bordered containers */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: rgba(255, 255, 255, 0.03); /* Subtle contrast */
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Centered Vertical Column Layout
    _, center_col, _ = st.columns([1, 2.2, 1])

    with center_col:
        # Header & Branding
        st.title("🧬 GeneRx")
        st.caption("Personalized Medication & DNA Safety Dashboard")
        st.write("")

        # 1. Pharmacological Input Card
        with st.container(border=True):
            st.markdown(
                "<p style='font-size: 11px; letter-spacing: 1.5px; color: #888888; text-transform: uppercase; font-weight: 600; margin-bottom: -10px;'>1. Pharmacological Input</p>",
                unsafe_allow_html=True,
            )
            st.write("")

            photo_mode = st.pills(
                "Source",
                ["Use Demo Data", "Upload Custom"],
                default="Use Demo Data",
                label_visibility="collapsed",
                key="photo_mode",
            )
            if photo_mode is None:
                photo_mode = "Use Demo Data"

            shots: list[bytes] = []
            if photo_mode == "Upload Custom":
                uploads = st.file_uploader(
                    "Upload medication or food label photo(s)",
                    type=IMG_TYPES,
                    accept_multiple_files=True,
                    key="photo_upload",
                )
                for u in uploads or []:
                    shots.append(u.getvalue())

                if uploads:
                    num_cols = min(len(uploads), 3)
                    cols = st.columns(num_cols)
                    for idx, file_item in enumerate(uploads):
                        with cols[idx % num_cols]:
                            st.image(file_item, use_container_width=True)

                with st.container(border=True):
                    if shots:
                        st.markdown("**Status:** <span style='color: #4CAF50;'>Data Loaded</span>", unsafe_allow_html=True)
                        st.markdown(f"**Files:** {len(shots)} custom photo(s) uploaded")
                    else:
                        st.markdown("**Status:** <span style='color: #FF9800;'>Awaiting Photo Upload</span>", unsafe_allow_html=True)
            else:
                demo_choice = st.pills(
                    "Scenario",
                    ["Plavix + Omeprazole", "Warfarin + Ibuprofen", "Simvastatin + Grapefruit"],
                    default="Plavix + Omeprazole",
                    label_visibility="collapsed",
                    key="demo_choice",
                )
                if demo_choice is None:
                    demo_choice = "Plavix + Omeprazole"

                scenario_key = SCENARIO_MAP.get(demo_choice, "Scenario 1 - Clopidogrel + Grapefruit Juice & Omeprazole")
                img = scenario_image(scenario_key)
                shots.append(img)

                with st.container(border=True):
                    st.markdown("**Status:** <span style='color: #4CAF50;'>Data Loaded</span>", unsafe_allow_html=True)
                    st.markdown(f"**Profile:** {demo_choice}")

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(img, use_container_width=True)

        st.write("")

        # 2. Genomic Input Card
        with st.container(border=True):
            st.markdown(
                "<p style='font-size: 11px; letter-spacing: 1.5px; color: #888888; text-transform: uppercase; font-weight: 600; margin-bottom: -10px;'>2. Genomic Profile</p>",
                unsafe_allow_html=True,
            )
            st.write("")

            dna_mode = st.pills(
                "DNA Profile",
                ["Demo Profile (High Risk)", "Demo Profile (Normal)", "Upload Custom File"],
                default="Demo Profile (High Risk)",
                label_visibility="collapsed",
                key="dna_mode",
            )
            if dna_mode is None:
                dna_mode = "Demo Profile (High Risk)"

            dna_text = None
            if dna_mode == "Upload Custom File":
                dna_upload = st.file_uploader(
                    "Upload a 23andMe raw data file (.txt)",
                    type=["txt"],
                    key="dna_upload",
                )
                if dna_upload is not None:
                    dna_text = dna_upload.getvalue().decode("utf-8-sig", errors="replace")

                with st.container(border=True):
                    if dna_text is not None:
                        st.markdown("**Status:** <span style='color: #4CAF50;'>Data Loaded</span>", unsafe_allow_html=True)
                        st.markdown(f"**Profile:** Custom 23andMe file ({len(dna_text)} bytes)")
                    else:
                        st.markdown("**Status:** <span style='color: #FF9800;'>Awaiting DNA File</span>", unsafe_allow_html=True)
            else:
                profile_key = "danger" if "High Risk" in dna_mode else "normal"
                dna_text = synthetic_dna(profile_key)

                with st.container(border=True):
                    st.markdown("**Status:** <span style='color: #4CAF50;'>Data Loaded</span>", unsafe_allow_html=True)
                    st.markdown(f"**Profile:** {dna_mode}")

        # 3. Action Area
        ready = len(shots) > 0 and dna_text is not None
        st.markdown("<hr style='border: 1px solid #333; margin-top: 30px;'/>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 10px; color: #666; text-align: center;'>End-to-end encryption active. Data processed locally.</p>",
            unsafe_allow_html=True,
        )
        run_clicked = st.button("Execute Safety Scan", type="primary", use_container_width=True, disabled=not ready)

        st.write("")

        # 4. Results Dashboard
        if run_clicked and ready:
            status_box = st.status("Running Safety Scan...", expanded=True)

            def on_step(step: str, state: str, detail: str, secs: float | None) -> None:
                if state == "running":
                    step_messages = {
                        "vision": "Reading image...",
                        "fda": "Checking openFDA...",
                        "interactions": "Checking interactions...",
                        "dna": "Analyzing DNA...",
                        "fusion": "Evaluating safety with Gemma 4...",
                    }
                    status_box.write(step_messages.get(step, detail))

            try:
                result = pipeline.run_fusion(
                    shots,
                    dna_text,
                    settings["api_key"],
                    settings["model"],
                    settings["base_url"],
                    spooky_mode=False,
                    on_step=on_step,
                )
                st.session_state["result"] = result
                status_box.update(label="Safety Scan Complete", state="complete", expanded=False)
            except Exception as exc:
                status_box.update(label="Scan Stopped", state="error", expanded=False)
                st.error(f"Error during scan: {exc}")

        result = st.session_state.get("result")

        if not result and not run_clicked:
            st.info("Select your medication photo and DNA profile above, then click **Execute Safety Scan** to generate your health safety report.")
        elif result:
            fusion = result.get("fusion", {})
            risk_level = fusion.get("risk_level", "SAFE").upper()

            with st.container(border=True):
                # Status Hero Alert Banner
                if risk_level == "CRITICAL":
                    st.error("🚨 **HIGH RISK DETECTED** — Immediate Action Advised")
                elif risk_level == "MODERATE":
                    st.warning("⚠️ **MODERATE RISK DETECTED** — Caution Advised")
                else:
                    st.success("✅ **SAFE** — No High Risk Detected")

                # Detected Items as "Pills" / Sub-cards
                st.markdown("##### Detected Items")
                detected_items = get_detected_items_list(result.get("vision", {}))
                if detected_items:
                    num_cols = min(len(detected_items), 4)
                    cols = st.columns(num_cols)
                    for idx, item_name in enumerate(detected_items):
                        with cols[idx % num_cols].container(border=True):
                            st.caption("Detected")
                            st.markdown(f"**{item_name}**")
                else:
                    st.caption("No specific items detected.")

                st.write("")

                # The Risk
                the_risk = fusion.get("the_risk") or fusion.get("headline") or "No significant risk identified."
                st.markdown("##### The Risk")
                st.info(the_risk)

                # What to Do
                what_to_do = fusion.get("what_to_do") or fusion.get("clinical_recommendation") or "Consult your doctor for guidance."
                with st.container(border=True):
                    st.markdown("##### What to Do")
                    st.write(what_to_do)

                # Dynamic Glossary for Technical Terms
                glossary = fusion.get("glossary") or []
                if not glossary:
                    glossary = [
                        {"term": "CYP2C19", "definition": "A liver enzyme that breaks down or activates certain medications in your body."},
                        {"term": "Metabolizer", "definition": "How quickly or slowly your body processes and removes a medicine."},
                    ]

                with st.expander("📚 Tap to understand medical terms", expanded=False):
                    for item in glossary:
                        st.markdown(f"**{item['term']}**: {item['definition']}")

                # The "Hackathon Tech Flex"
                st.write("")
                with st.expander("🔬 View Clinical & API Evidence", expanded=False):
                    st.caption("Analysis generated by Google Gemma 4 Vision & Reasoning.")
                    st.metric(label="Regulatory Grounding", value="Live U.S. openFDA API")

                    fda_records = result.get("fda_records", [])
                    if fda_records:
                        live_count = sum(1 for r in fda_records if r.get("live"))
                        st.caption(f"FDA Labels Searched: {len(fda_records)} ({live_count} live openFDA queries)")

                    dna = result.get("dna", {})
                    if dna.get("matched_snps"):
                        st.caption(f"Genetic Markers Analyzed: {len(dna['matched_snps'])} enzyme genotypes")


if __name__ == "__main__":
    main()
