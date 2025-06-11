# SonoVLM: A Generalist Ultrasound Vision-Language Model  


A multimodal AI system for ultrasound analysis with capabilities in cross-organ understanding, abnormality detection, diagnostic reasoning, structured reporting, and patient-centric dialogue.

---

## 🔥 Latest News  
- **2025/05/24**: 🎉 Official repository launched!  
- **2025/06/10**: Code is now available
> 📌 **Code is now publicly available** 

---

## 🧠 Key Features  
- **Multi-Organ Analysis**: 7 anatomical structures supported (e.g., thyroid, breast, liver, heart, kidney)  
- **Disease Diagnosis**: Ultrasound-based classification
- **Diagnostic Reasoning**:  disease differentiation 
- **Ultrasound Reporting**: Automatic structured report generation 
- **Clinical Dialogue**: Multi-turn conversation support for doctor-patient-system interaction

---
## 🔥 Demo  
<video controls>
  <source src="VLM演示.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
## 📈 Technical Architecture  
pass

## 🚀 Getting Started  
**Note**: Code will be released publicly after publication of our paper.  
<details>
<summary><b>1. Installation (Coming Soon) </b></summary>
pass
</details>
<details>
<summary><b>2. Prepare your finetuning data</b></summary>

Like LLaVA, we anticipate that the data will reside within a JSON file, composed of a collection of dictionaries. In this structure, each individual dictionary corresponds to a distinct sample.
```json
   [
    {
        "id": "215168",
        "system_prompt": "You are a helpful assistant.",
        "image": [
            "215168_1.jpeg",
            "215168_2.jpeg"
        ],
        "description": "双侧乳腺腺体结构稍紊乱，乳导管不扩张，双侧乳腺未见明确占位性病变。双侧腋下未见明显肿大淋巴结。",
        "conversations": [
            {
                "from": "human",
                "value": "<image><image>Based on the ultrasound image, can you speculate on the functional status of the examined organ?"
            },
            {
                "from": "gpt",
                "value": "双侧乳腺腺体结构稍紊乱，乳导管不扩张，双侧乳腺未见明确占位性病变。双侧腋下未见明显肿大淋巴结。"
            },
        ]
    }
]
```
</details>
<details>
<summary><b>3. Perform finetuning</b></summary>
   
```
deepspeed train.py 
```
You can modify the parameter settings as needed, such as 
   ```
   deepspeed train.py
--per_device_train_batch_size 16
```
