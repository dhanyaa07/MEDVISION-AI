import os
import uuid
import cv2
import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle, PageBreak
)
from xml.sax.saxutils import escape

# ==========================================
# 1. LOAD EXISTING MODEL
# ==========================================

MODEL_PATH = os.getenv("MEDVISION_MODEL_PATH", os.path.join(os.path.dirname(__file__), "models", "MEDVISION_AI_FINAL_V2.keras"))

mv_model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# ==========================================
# 2. EXISTING GRAD-CAM
# ==========================================

mv_base = mv_model.get_layer("efficientnetb0")

mv_feature_model = tf.keras.Model(
    inputs=mv_base.input,
    outputs=[
        mv_base.get_layer("top_conv").output,
        mv_base.output
    ]
)

mv_pool = mv_model.get_layer("global_average_pooling2d")
mv_drop1 = mv_model.get_layer("dropout")
mv_dense = mv_model.get_layer("dense")
mv_drop2 = mv_model.get_layer("dropout_1")
mv_classifier = mv_model.get_layer("dense_1")


def make_gradcam(image):

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    with tf.GradientTape() as tape:

        conv_output, features = mv_feature_model(
            image, training=False
        )

        x = mv_pool(features)
        x = mv_drop1(x, training=False)
        x = mv_dense(x)
        x = mv_drop2(x, training=False)

        predictions = mv_classifier(x)

        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    gradients = tape.gradient(loss, conv_output)

    if gradients is None:
        raise ValueError("Grad-CAM calculation failed.")

    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(
        conv_output[0] * weights, axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), predictions.numpy()[0]


# ==========================================
# 3. MEDICAL REASONING KNOWLEDGE BASE
# ==========================================

MEDICAL_INFO = {

    "glioma": {

        "anatomy":
        "Gliomas arise in glial cells, which provide structural "
        "and functional support to neurons in the brain or spinal cord.",

        "pathophysiology":
        "Gliomas develop through abnormal proliferation of glial-lineage "
        "cells. Their biological behaviour varies considerably according "
        "to molecular alterations, histological type and grade.",

        "causes":
        "Most gliomas have no identifiable individual cause. Genetic "
        "alterations within tumor cells are involved in their development. "
        "Recognized risk factors include previous ionizing radiation "
        "exposure. Some inherited cancer-predisposition syndromes are "
        "associated with increased risk.",

        "symptoms":
        "Depending on the affected brain region, manifestations may "
        "include persistent headaches, seizures, weakness, speech "
        "difficulties, visual disturbances, memory changes and "
        "neurological deficits.",

        "complications":
        "Depending on tumour type, grade and location, possible "
        "complications include cerebral edema, seizures, increased "
        "intracranial pressure and neurological impairment.",

        "differential":
        "Other intracranial abnormalities that may require consideration "
        "include metastases, primary CNS lymphoma, abscesses and other "
        "brain lesions. Imaging appearance alone may not reliably "
        "distinguish all these conditions.",

        "evaluation":
        "Specialist evaluation may include MRI with appropriate "
        "sequences and contrast, neurological examination, advanced "
        "imaging when indicated, and tissue diagnosis with molecular "
        "testing where clinically appropriate.",

        "prognosis":
        "Prognosis varies substantially with molecular subtype, "
        "histological grade, patient factors, location and response "
        "to treatment. A four-class image classifier cannot estimate "
        "individual prognosis."
    },

    "meningioma": {

        "anatomy":
        "Meningiomas originate from cells associated with the meninges, "
        "the protective membranes surrounding the brain and spinal cord.",

        "pathophysiology":
        "Meningiomas develop through abnormal proliferation of "
        "meningeal cells. Many grow slowly, although their growth "
        "rate and biological behaviour vary.",

        "causes":
        "Most cases do not have a clearly identifiable individual cause. "
        "Recognized associations include previous radiation exposure, "
        "certain genetic alterations and inherited conditions such "
        "as neurofibromatosis type 2.",

        "symptoms":
        "Symptoms depend on tumour size and location and may include "
        "headaches, seizures, visual disturbances, hearing changes, "
        "weakness or other neurological symptoms. Some meningiomas "
        "are discovered incidentally without symptoms.",

        "complications":
        "Depending on location and growth, possible complications "
        "include compression of adjacent brain tissue, cerebral edema, "
        "seizures and neurological dysfunction.",

        "differential":
        "Other lesions that may resemble meningiomas include dural "
        "metastases, solitary fibrous tumours and other extra-axial "
        "lesions. A specialist considers the full imaging appearance.",

        "evaluation":
        "Evaluation may involve MRI with contrast, neurological "
        "assessment and review of lesion location and growth over time. "
        "Histopathological examination may be required when treatment "
        "or surgery is undertaken.",

        "prognosis":
        "Prognosis depends on histological grade, tumour location, "
        "extent of removal, recurrence and other clinical factors. "
        "The current classifier cannot determine tumour grade "
        "or predict an individual outcome."
    },

    "notumor": {

        "anatomy":
        "The brain contains multiple anatomical structures including "
        "the cerebral hemispheres, cerebellum, brainstem and ventricular "
        "system.",

        "pathophysiology":
        "The no-tumor category represents images labelled as notumor "
        "in the training dataset. It is a classification category, "
        "not a comprehensive assessment of brain health.",

        "causes":
        "A no-tumor prediction does not identify the cause of any "
        "symptoms. Headaches and neurological symptoms may have "
        "many unrelated causes.",

        "symptoms":
        "Symptoms cannot be established or excluded through this "
        "classification. Clinical symptoms require independent "
        "history-taking and medical evaluation.",

        "complications":
        "The model's no-tumor classification does not rule out "
        "non-neoplastic abnormalities, subtle lesions or other "
        "neurological conditions.",

        "differential":
        "Other conditions that may require clinical assessment include "
        "vascular abnormalities, inflammatory conditions, infections, "
        "developmental abnormalities and non-tumour structural changes.",

        "evaluation":
        "If symptoms or clinical concerns persist, a qualified "
        "radiologist should interpret the complete MRI examination "
        "and recommend further evaluation where appropriate.",

        "prognosis":
        "A no-tumor model prediction cannot establish a person's "
        "overall neurological health or predict clinical outcome."
    },

    "pituitary": {

        "anatomy":
        "The pituitary gland is a small endocrine gland located at "
        "the base of the brain within the sella turcica. It is "
        "connected to the hypothalamus and regulates several "
        "important hormonal functions.",

        "pathophysiology":
        "Pituitary adenomas arise from pituitary cells and may "
        "produce excess hormones or cause hormonal deficiencies "
        "through compression of normal gland tissue. Their behaviour "
        "depends on cell type, hormone secretion, size and location.",

        "causes":
        "Most pituitary adenomas occur sporadically without an "
        "identifiable individual cause. Some are associated with "
        "inherited conditions, including multiple endocrine "
        "neoplasia type 1 and other familial syndromes. The "
        "individual cause cannot be established from this classifier.",

        "symptoms":
        "Possible manifestations include headaches, visual field "
        "disturbances, menstrual irregularities, galactorrhea, "
        "changes in libido, fatigue and symptoms associated with "
        "excessive or deficient hormone production. Symptoms "
        "vary according to the lesion and hormone profile.",

        "complications":
        "Depending on the lesion, possible complications include "
        "hormonal imbalance, compression of nearby structures "
        "such as the optic chiasm and, rarely, acute complications "
        "such as pituitary apoplexy.",

        "differential":
        "Pituitary-region abnormalities include adenomas, Rathke "
        "cleft cysts, craniopharyngiomas, hypophysitis and other "
        "sellar or parasellar lesions. Dedicated imaging and "
        "clinical assessment are needed to distinguish them.",

        "evaluation":
        "A specialist may consider dedicated pituitary MRI with "
        "appropriate sequences, hormonal investigations such as "
        "prolactin and pituitary-axis testing, and formal visual "
        "field assessment when clinically indicated.",

        "prognosis":
        "Prognosis varies according to lesion type, hormone "
        "secretion, size, extension, effects on surrounding "
        "structures and response to treatment. The current "
        "classifier cannot determine any of these individual "
        "prognostic factors."
    }
}


# ==========================================
# 4. IMAGE QUALITY ANALYSIS
# ==========================================

def analyze_image_quality(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "minimum": int(np.min(gray)),
        "maximum": int(np.max(gray))
    }


# ==========================================
# 5. EXPANDED REPORT GENERATOR
# ==========================================

def generate_report(data):

    info = MEDICAL_INFO[data["prediction"]]

    ranked = sorted(
        zip(CLASS_NAMES, data["probabilities"]),
        key=lambda x: x[1],
        reverse=True
    )

    margin = (ranked[0][1] - ranked[1][1]) * 100

    quality = data["quality"]

    report = f"""
MEDVISION AI - COMPREHENSIVE MRI ANALYSIS REPORT

Report ID: {data["report_id"]}
Date: {data["date"]}
Image: {data["filename"]}

==================================================
1. AI CLASSIFICATION RESULTS
==================================================

Predicted Category: {data["prediction"].upper()}
Model Confidence Score: {data["confidence"]:.2f}%

The predicted category is the class with the highest
model output score. This is an AI classification result,
not a confirmed medical diagnosis.

==================================================
2. CLASS PROBABILITY DISTRIBUTION
==================================================

"""

    for name, probability in ranked:
        report += f"{name.upper()}: {probability * 100:.2f}%\n"

    report += f"""
Top Two Class Score Difference: {margin:.2f} percentage points.

Softmax scores are not calibrated medical probabilities.

==================================================
3. MRI IMAGE INFORMATION
==================================================

Filename: {data["filename"]}
Image Format: {data["format"]}
Original Dimensions: {data["width"]} x {data["height"]}
Color Mode: RGB
Model Input Dimensions: 224 x 224

Mean Grayscale Intensity: {quality["brightness"]:.2f}
Intensity Standard Deviation: {quality["contrast"]:.2f}
Laplacian Sharpness Variance: {quality["sharpness"]:.2f}
Minimum Pixel Intensity: {quality["minimum"]}
Maximum Pixel Intensity: {quality["maximum"]}

==================================================
4. IMAGE-BASED OBSERVATIONS
==================================================

The image has been converted into RGB format and
resized to 224 x 224 pixels for classification.

The measured brightness, contrast and sharpness
describe technical image characteristics.

These measurements cannot establish tumour boundaries,
dimensions, biological behaviour or anatomical location.

==================================================
5. GRAD-CAM EXPLAINABILITY
==================================================

Grad-CAM highlights image regions that contributed
to the predicted classification.

Mean Grad-CAM Activation: {data["cam_mean"]:.4f}
Maximum Grad-CAM Activation: {data["cam_max"]:.4f}
Highlighted Activation Area: {data["cam_area"]:.2f}%

The highlighted activation area is not a tumour
area measurement or segmentation result.

==================================================
6. ANATOMICAL INFORMATION
==================================================

{info["anatomy"]}

==================================================
7. PATHOPHYSIOLOGY
==================================================

{info["pathophysiology"]}

This is general medical background, not a biological
finding established from the uploaded image.

==================================================
8. POSSIBLE CAUSES AND RISK FACTORS
==================================================

{info["causes"]}

The current AI classifier cannot identify the cause
of an individual patient's condition.

==================================================
9. ASSOCIATED CLINICAL MANIFESTATIONS
==================================================

{info["symptoms"]}

Symptoms cannot be confirmed or excluded from
image classification alone.

==================================================
10. POTENTIAL COMPLICATIONS
==================================================

{info["complications"]}

These are possible complications associated with
relevant medical conditions, not complications
identified in the uploaded MRI.

==================================================
11. DIFFERENTIAL CONSIDERATIONS
==================================================

{info["differential"]}

These conditions are general differential considerations.
The model does not perform differential diagnosis.

==================================================
12. DIAGNOSTIC EVALUATION
==================================================

{info["evaluation"]}

==================================================
13. PROGNOSTIC CONSIDERATIONS
==================================================

{info["prognosis"]}

Individual prognosis cannot be estimated using
the current four-class image classification model.

==================================================
14. MODEL ARCHITECTURE
==================================================

Architecture: EfficientNetB0
Learning Approach: Transfer Learning
Input Dimensions: 224 x 224
Input Channels: RGB
Classification Classes: 4
Output Activation: Softmax

==================================================
15. MODEL EVALUATION
==================================================

Validation Accuracy: 90.99%
Macro Precision: 91.13%
Macro Recall: 90.99%
Macro F1-Score: 90.98%
Weighted F1-Score: 90.98%

==================================================
16. RESEARCH LIMITATIONS
==================================================

- Validation uses a group-aware image-level split.
- Patient-level independent validation is not established.
- Dataset representativeness is not guaranteed.
- Softmax scores are not calibrated probabilities.
- Grad-CAM is an approximate visual explanation.
- No tumour segmentation or physical measurements
  are performed.
- Clinical history and laboratory results are unavailable.
- Medical explanations are general knowledge.
- The system is not clinically validated.

==================================================
17. MEDICAL DISCLAIMER
==================================================

MEDVISION AI is an academic AI research prototype.

It is not intended for medical diagnosis or treatment.
Its classification must not replace professional
radiological interpretation or clinical evaluation.

Generated by MEDVISION AI
"""

    return report


# ==========================================
# 6. MRI ANALYSIS FUNCTION
# ==========================================

def analyze_mri(image):

    if image is None:
        raise gr.Error("Please upload an MRI image.")

    try:

        original = image.convert("RGB")
        original_array = np.array(original)

        resized = cv2.resize(original_array, (224, 224))

        input_image = np.expand_dims(
            resized.astype(np.float32), axis=0
        )

        heatmap, probabilities = make_gradcam(input_image)

        predicted_index = int(np.argmax(probabilities))
        prediction = CLASS_NAMES[predicted_index]

        confidence = float(
            probabilities[predicted_index] * 100
        )

        scores = {
            name: float(probability)
            for name, probability in zip(
                CLASS_NAMES, probabilities
            )
        }

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)

        colored_heatmap = cv2.applyColorMap(
            heatmap, cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            resized, 0.6, colored_heatmap, 0.4, 0
        )

        overlay = cv2.cvtColor(
            overlay, cv2.COLOR_BGR2RGB
        )

        quality = analyze_image_quality(original_array)

        normalized_heatmap = heatmap.astype(np.float32) / 255.0

        cam_mean = float(np.mean(normalized_heatmap))
        cam_max = float(np.max(normalized_heatmap))

        cam_area = float(
            np.mean(normalized_heatmap >= 0.5) * 100
        )

        report_id = uuid.uuid4().hex[:8].upper()

        filename = getattr(image, "filename", None) or "Uploaded MRI"

        result_data = {
            "report_id": report_id,
            "date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "filename": filename,
            "format": original.format or "RGB image",
            "width": original.width,
            "height": original.height,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "quality": quality,
            "cam_mean": cam_mean,
            "cam_max": cam_max,
            "cam_area": cam_area,
            "original": original_array,
            "gradcam": overlay
        }

        report = generate_report(result_data)

        return scores, overlay, report, result_data

    except Exception as e:
        raise gr.Error(f"MRI analysis failed: {str(e)}")


# ==========================================
# 7. PDF REPORT GENERATOR
# ==========================================

def generate_pdf(data):

    if data is None:
        raise gr.Error("Please analyze an MRI image first.")

    try:

        report_id = data["report_id"]

        pdf_path = f"/content/MEDVISION_AI_Report_{report_id}.pdf"

        original_path = f"/content/mv_original_{report_id}.png"
        gradcam_path = f"/content/mv_gradcam_{report_id}.png"
        matrix_path = f"/content/mv_matrix_{report_id}.png"

        Image.fromarray(data["original"]).save(original_path)
        Image.fromarray(data["gradcam"]).save(gradcam_path)

        cm = np.array([
            [243, 35, 0, 3],
            [13, 235, 3, 29],
            [1, 2, 271, 6],
            [0, 8, 1, 271]
        ])

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_NAMES, rotation=25)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title("Validation Confusion Matrix")

        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.tight_layout()
        plt.savefig(matrix_path, dpi=180)
        plt.close(fig)

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            rightMargin=45,
            leftMargin=45,
            topMargin=45,
            bottomMargin=45
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "MainTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            fontSize=21,
            spaceAfter=12
        )

        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#17365D"),
            spaceBefore=10,
            spaceAfter=6
        )

        story = [
            Paragraph("MEDVISION AI", title_style),
            Paragraph(
                "Comprehensive Brain MRI Analysis Report",
                ParagraphStyle(
                    "subtitle",
                    parent=styles["Normal"],
                    alignment=TA_CENTER
                )
            ),
            Spacer(1, 20),
            Paragraph("CLASSIFICATION RESULTS", heading_style),
            Paragraph(
                f'<b>Predicted Category:</b> {escape(data["prediction"].upper())}',
                styles["BodyText"]
            ),
            Paragraph(
                f'<b>Model Confidence:</b> {data["confidence"]:.2f}%',
                styles["BodyText"]
            )
        ]

        probability_table = [["Class", "Model Score"]]

        for name, probability in zip(
            CLASS_NAMES, data["probabilities"]
        ):
            probability_table.append([
                name.upper(), f"{probability * 100:.2f}%"
            ])

        table = Table(probability_table, colWidths=[220, 220])

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8)
        ]))

        story.append(table)
        story.append(Spacer(1, 15))

        story.append(
            Paragraph("DETAILED MEDICAL REASONING", heading_style)
        )

        report = generate_report(data)

        for line in report.splitlines():

            line = line.strip()

            if not line:
                story.append(Spacer(1, 3))

            elif line.startswith("="):
                continue

            elif line.isupper() and len(line) < 100:
                story.append(
                    Paragraph(escape(line), heading_style)
                )

            else:
                story.append(
                    Paragraph(escape(line), styles["BodyText"])
                )

        story.append(PageBreak())

        story.append(
            Paragraph("ORIGINAL MRI IMAGE", title_style)
        )

        story.append(
            RLImage(original_path, width=4.7*inch, height=3.8*inch,
                    kind="proportional")
        )

        story.append(Spacer(1, 15))

        story.append(
            Paragraph("GRAD-CAM VISUALIZATION", title_style)
        )

        story.append(
            RLImage(gradcam_path, width=4.7*inch, height=3.8*inch,
                    kind="proportional")
        )

        story.append(PageBreak())

        story.append(
            Paragraph("MODEL VALIDATION EVALUATION", title_style)
        )

        metrics = [
            ["Metric", "Validation Result"],
            ["Accuracy", "90.99%"],
            ["Macro Precision", "91.13%"],
            ["Macro Recall", "90.99%"],
            ["Macro F1-Score", "90.98%"],
            ["Weighted F1-Score", "90.98%"]
        ]

        table = Table(metrics, colWidths=[220, 220])

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8)
        ]))

        story.append(table)

        story.append(
            Paragraph("CLASS-WISE PERFORMANCE", heading_style)
        )

        class_metrics = [
            ["Class", "Precision", "Recall", "F1", "Support"],
            ["Glioma", "94.92%", "86.48%", "90.50%", "281"],
            ["Meningioma", "83.04%", "83.93%", "83.48%", "280"],
            ["No Tumor", "97.13%", "96.79%", "96.96%", "280"],
            ["Pituitary", "89.44%", "96.79%", "92.97%", "280"]
        ]

        table = Table(class_metrics, colWidths=[100, 85, 85, 85, 65])

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7)
        ]))

        story.append(table)

        story.append(
            Paragraph("VALIDATION CONFUSION MATRIX", heading_style)
        )

        story.append(
            RLImage(matrix_path, width=4.8*inch, height=3.8*inch)
        )

        story.append(Spacer(1, 10))

        story.append(
            Paragraph(
                "The validation metrics describe the model-level "
                "evaluation and are not individual patient results.",
                styles["BodyText"]
            )
        )

        doc.build(story)

        for path in [original_path, gradcam_path, matrix_path]:
            if os.path.exists(path):
                os.remove(path)

        return pdf_path

    except Exception as e:
        raise gr.Error(f"PDF generation failed: {str(e)}")


# ==========================================
# 8. EXISTING GRADIO INTERFACE
# ==========================================

with gr.Blocks(title="MEDVISION AI") as mv_demo:

    gr.Markdown("""
    # MEDVISION AI
    ### Brain MRI Classification and Explainable AI System

    Upload a brain MRI image to generate an AI classification,
    Grad-CAM visualization and comprehensive medical analysis report.
    """)

    mv_state = gr.State(None)

    with gr.Row():

        with gr.Column():

            mv_input = gr.Image(
                label="Upload Brain MRI",
                type="pil"
            )

            mv_analyze = gr.Button(
                "Analyze MRI",
                variant="primary"
            )

        with gr.Column():

            mv_prediction = gr.Label(
                label="Classification Results",
                num_top_classes=4
            )

            mv_gradcam = gr.Image(
                label="Grad-CAM Visualization"
            )

    mv_report = gr.Textbox(
        label="Comprehensive MRI Analysis Report",
        lines=22,
        interactive=False
    )

    mv_analyze.click(
        fn=analyze_mri,
        inputs=mv_input,
        outputs=[
            mv_prediction,
            mv_gradcam,
            mv_report,
            mv_state
        ]
    )

    gr.Markdown("### Download Your Analysis")

    mv_pdf_button = gr.Button(
        "Generate & Download PDF Report",
        variant="primary"
    )

    mv_pdf_output = gr.File(
        label="Download MEDVISION AI Report"
    )

    mv_pdf_button.click(
        fn=generate_pdf,
        inputs=mv_state,
        outputs=mv_pdf_output
    )

    gr.Markdown("""
    **Disclaimer:** MEDVISION AI is an academic research prototype,
    not a clinically validated diagnostic system.
    """)

mv_demo.launch(share=os.getenv("GRADIO_SHARE", "false").lower() == "true", show_error=True)