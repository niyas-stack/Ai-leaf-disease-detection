import os
import numpy as np
from torchsummary import summary
import torch
import PIL
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import base64
# Load the model
model = torchvision.models.resnet18(pretrained=True)
classes = dict({0:'The above leaf is Cassava (Cassava Mosaic) ', 
                1:'The above leaf is Cassava CB (Cassava Bacterial Blight)', 
                2:'The above leaf is Cassava Healthy leaf', 
                3:'This is not trained yet',
                4:'The above leaf is Tomato Bacterial spot', 
                5:'The above leaf is Tomato early blight',
                6:'The above leaf is Tomato Late blight',
                7:'The above leaf is Tomato Leaf Mold', 
                8:'The above leaf is Tomato Septoria leaf spot',
                9:'The above leaf is Tomato Spider mites Two-spotted spider mite', 
                10:'The above leaf is Tomato Target Spot',
                11:'The above leaf is Tomato Yellow Leaf Curl Virus', 
                12:'The above leaf is Tomato mosaic virus', 
                13:' The above leaf is Tomato healthy', 
                14:'The above leaf is bean angular leaf spot',
                15:'The above leaf is bean healthy', 
                16:'The above leaf is bean rust'})

remedies = {
    'The above leaf is Cassava (Cassava Mosaic) ': [
         'Use of resistant variety Sripadmanaba suited for Tamil Nadu and Kerala. Mosaic tolerant varieties such as H-97 may be used to minimize economic loss of tubers. Select setts from healthy plants. Roug out and destroy infected plants in the field at early stage.Control whitefly by installing yellow sticky traps, removal of weed hosts, spray neem oil (20 ml / litre of water). Spray Dimethoate 30 EC (2 ml / litre of water) to control the vector.', 'കിഴങ്ങുവർഗ്ഗങ്ങളുടെ സാമ്പത്തിക നഷ്ടം കുറയ്ക്കുന്നതിന് H-97 പോലുള്ള മൊസൈക്ക് സഹിഷ്ണുതയുള്ള ഇനങ്ങൾ ഉപയോഗിക്കാം. ആരോഗ്യമുള്ള ചെടികളിൽ നിന്ന് സെറ്റുകൾ തിരഞ്ഞെടുക്കുക. രോഗബാധയക്ക് ആദ്യത്തെ സ്റ്റേജിൽ കിട്ടുന്ന അമൃതം വെള്ളിയിൽ കൊണ്ടുവരുന്നതായിരിക്കാം, അതിനാൽ വെട്ടിച്ചെടികൾ നന്നായിരിക്കണം. വെള്ളിയെക്കുറിച്ചുള്ള പിഴച്ചെടികൾ നീങ്ങിനിർത്തുക, വിറക് ഓയില് (20 മി.എല് കൂട്ടിച്ചേർക്കുക) സ്പ്രേ ചെയ്യുക. വെട്ടിച്ചെടികൾ നന്നായിരിക്കണം. വൈറ്റ്ഫ്ലൈയെ നിയന്ത്രിക്കുന്നതിനും വളർത്തിപ്പാട്ടും വീഴ്ചയും വേർതിരിക്കുക.',
         'CASSAVA(MOSAIC)(ENG).mp3', 'CASSAVA(MOSAIC)(MAL).m4a'
    ],
    'The above leaf is Cassava CB (Cassava Bacterial Blight)': [
       'Chemical control is not effective. Use of resistant or tolerant varieties is the only method of control. Among improved varieties, H-97, H-226, H-1687 and H-2304 are tolerant to the disease while H-165 is highly susceptible. Among the local varieties, M-4 is tolerant to the disease.', 'രാസ നിയന്ത്രണം ഫലപ്രദമല്ല. പ്രതിരോധശേഷിയുള്ളതോ സഹിഷ്ണുതയുള്ളതോ ആയ ഇനങ്ങളുടെ ഉപയോഗം മാത്രമാണ് നിയന്ത്രണ മാർഗ്ഗം. മെച്ചപ്പെട്ട ഇനങ്ങളിൽ, H-97, H-226, H-1687, H-2304 എന്നിവ രോഗത്തെ സഹിഷ്ണുതയുള്ളവയാണ്, അതേസമയം H-165 വളരെ രോഗസാധ്യതയുള്ളവയാണ്. പ്രാദേശിക ഇനങ്ങളിൽ, M-4 രോഗത്തോട് സഹിഷ്ണുത പുലർത്തുന്നു',
       'cassava bacterial blight eng.mp3', 'cassava bacterialblight mal.m4a'
    ],
     'The above leaf is Tomato Bacterial spot': [
       'Hot water treatment of seeds at 50°C for 25 minutes is effective. Crop rotation with non-host crop helps in reducing the disease incidence. Soaking of seed in solution of Agrimycin-100 (1 gram / 10 litre of water) or Streptocycline (0.5 gram/litre of water) or with Pseudomonas fluorescens (20 gram/litre of water / kg of seed) for 30 minute protects the seedlings in the initial stages of growth.Soil application of Pseudomonas fluorescens fortified in neem cake also reduces the disease incidence. Affected plants should be removed and destroyed. Disinfect the area with bleaching powder @ 15 kg / ha in endemic fields.The planting pits should be prophylatically drenched with Copper oxychloride 50 WP (2 gram/litre of water) or Copper hydroxide 77 WP (1 gram/litre of water) two weeks before planting. Soil drench Agrimycin-100 (1 gram / 10 litre of water) or Streptocycline (0.5 gram/litre of water) when the disease is noticed.', 'വിത്ത് 50 ഡിഗ്രി സെൽഷ്യസിൽ 25 മിനിറ്റ് ചൂടുവെള്ളത്തിൽ ശുദ്ധീകരിക്കുന്നത് ഫലപ്രദമാണ്. ആതിഥേയമല്ലാത്ത വിളകൾ ഉപയോഗിച്ച് വിള ഭ്രമണം ചെയ്യുന്നത് രോഗബാധ കുറയ്ക്കാൻ സഹായിക്കുന്നു. വിത്ത് അഗ്രിമൈസിൻ-100 (1 ഗ്രാം / 10 ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ സ്ട്രെപ്റ്റോസൈക്ലിൻ (0.5 ഗ്രാം/ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ സ്യൂഡോമോണസ് ഫ്ലൂറസെൻസ് (20 ഗ്രാം/ലിറ്റർ വെള്ളം / കിലോ വിത്ത്) എന്നിവയിൽ 30 മിനിറ്റ് മുക്കിവയ്ക്കുന്നത് തൈകളെ സംരക്ഷിക്കുന്നു. വളർച്ചയുടെ പ്രാരംഭ ഘട്ടത്തിൽ.വേപ്പിൻ പിണ്ണാക്ക് ഉറപ്പിച്ച സ്യൂഡോമോണസ് ഫ്ലൂറസെൻസ് മണ്ണിൽ പ്രയോഗിക്കുന്നതും രോഗബാധ കുറയ്ക്കുന്നു. ബാധിച്ച ചെടികൾ നീക്കം ചെയ്യുകയും നശിപ്പിക്കുകയും വേണം. എൻഡമിക് വയലുകളിൽ ഹെക്ടറിന് 15 കി.ഗ്രാം ബ്ലീച്ചിംഗ് പൗഡർ ഉപയോഗിച്ച് പ്രദേശം അണുവിമുക്തമാക്കുക.നടുന്നതിന് രണ്ടാഴ്ച മുമ്പ്, നടീൽ കുഴികളിൽ കോപ്പർ ഓക്സിക്ലോറൈഡ് 50 WP (2 ഗ്രാം/ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ കോപ്പർ ഹൈഡ്രോക്സൈഡ് 77 WP (1 ഗ്രാം/ലിറ്റർ വെള്ളം) ഉപയോഗിച്ച് നനയ്ക്കണം. രോഗം ശ്രദ്ധയിൽപ്പെട്ടാൽ അഗ്രിമൈസിൻ-100 (1 ഗ്രാം / 10 ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ സ്ട്രെപ്റ്റോസൈക്ലിൻ (0.5 ഗ്രാം/ലിറ്റർ വെള്ളം) മണ്ണിൽ നനയ്ക്കുക.',
       'tm bct spt.mp3', 'tomato bacterial spot(mal).m4a'
    ],
     'The above leaf is Tomato early blight': [
       'To manage Tomato Early Blight, it is recommended to implement cultural practices such as crop rotation, proper spacing of plants to enhance air circulation, and regular removal and destruction of infected plant debris. Fungicide applications may be necessary in severe cases. It is advisable to consult with local agricultural extension services or plant pathology experts for specific guidance on managing Early Blight in your region.Use of disease free seed tubers for raising the crop. The crop must be given balanced doses of fertilizers, especially nitrogen. Follow crop rotation. Removal and destruction of the affected plant parts.Spray Mancozeb 75 WP (3-4 gram/litre of water) or Carbendazim 50 WP (1 gram/litre of water). Solanaceous crops, which act as the collateral hosts for the pathogen, should not be cultivated in and around tomato fields.', 'വിള ഭ്രമണം, വായു സഞ്ചാരം വർധിപ്പിക്കാൻ ചെടികളുടെ ശരിയായ അകലം, രോഗം ബാധിച്ച ചെടികളുടെ അവശിഷ്ടങ്ങൾ പതിവായി നീക്കം ചെയ്യുകയും നശിപ്പിക്കുകയും ചെയ്യുന്നു. കഠിനമായ കേസുകളിൽ കുമിൾനാശിനി പ്രയോഗങ്ങൾ ആവശ്യമായി വന്നേക്കാം.വിള വളർത്തുന്നതിന് രോഗബാധയില്ലാത്ത വിത്ത് കിഴങ്ങുകൾ ഉപയോഗിക്കുക. വിളകൾക്ക് സമീകൃതമായ അളവിൽ വളങ്ങൾ നൽകണം, പ്രത്യേകിച്ച് നൈട്രജൻ. വിള ഭ്രമണം പിന്തുടരുക. ബാധിച്ച ചെടിയുടെ ഭാഗങ്ങൾ നീക്കം ചെയ്യുകയും നശിപ്പിക്കുകയും ചെയ്യുക. മാങ്കോസെബ് 75 WP (3-4 ഗ്രാം/ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ കാർബൻഡാസിം 50 WP (1 ഗ്രാം/ലിറ്റർ വെള്ളം) തളിക്കുക. രോഗാണുക്കൾക്ക് കൊളാറ്ററൽ ഹോസ്റ്റുകളായി പ്രവർത്തിക്കുന്ന സോളനേഷ്യസ് വിളകൾ തക്കാളി വയലുകളിലും പരിസരത്തും കൃഷി ചെയ്യാൻ പാടില്ല',
       'tmt earl blight.mp3', 'tmt ear blight(mal).m4a'
     ],
     'The above leaf is Tomato Late blight': [
       'Follow crop rotation, avoid planting tomato near potato fields, provide drainage, avoid flooding and use disease-free seeds and transplants. The seed material should be obtained from a disease free area. Before planting the seeds should be treated with Thiram 75 WS (3-4 gram/litre of water / kg of seed) or Mancozeb 75 WP (3-4 gram/litre of water / kg of seed).All blighted leaves must be removed and buried deep in the soil. Spray Mancozeb 75 WP (3-4 gram/litre of water) before the appearance of the disease is effective. Spray potassium phosphonate 40 SC (3-4 ml/litre of water) or Metalaxyl + Mancozeb (8+64) WP or Fosetly Al 80 WP (2 gram/litre of water) once the disease is noticed.', 'വിള ഭ്രമണം പിന്തുടരുക, ഉരുളക്കിഴങ്ങ് വയലുകൾക്ക് സമീപം തക്കാളി നടുന്നത് ഒഴിവാക്കുക, ഡ്രെയിനേജ് നൽകുക, വെള്ളപ്പൊക്കം ഒഴിവാക്കുക, രോഗബാധയില്ലാത്ത വിത്തുകളും പറിച്ചുനടലുകളും ഉപയോഗിക്കുക. രോഗബാധയില്ലാത്ത പ്രദേശത്തുനിന്നായിരിക്കണം വിത്ത് വസ്തു ലഭിക്കേണ്ടത്. വിത്ത് നടുന്നതിന് മുമ്പ് തിറം 75 WS (3-4 ഗ്രാം / ലിറ്റർ വെള്ളം / കിലോഗ്രാം വിത്ത്) അല്ലെങ്കിൽ മാങ്കോസെബ് 75 WP (3-4 ഗ്രാം / ലിറ്റർ വെള്ളം / കിലോ വിത്ത്) ഉപയോഗിച്ച് സംസ്കരിക്കണം.കരിഞ്ഞുണങ്ങിയ എല്ലാ ഇലകളും നീക്കം ചെയ്യുകയും മണ്ണിൽ ആഴത്തിൽ കുഴിച്ചിടുകയും വേണം. രോഗം പ്രത്യക്ഷപ്പെടുന്നതിന് മുമ്പ് മാങ്കോസെബ് 75 WP (3-4 ഗ്രാം/ലിറ്റർ വെള്ളത്തിൽ) തളിക്കുക. രോഗം ശ്രദ്ധയിൽപ്പെട്ടാൽ പൊട്ടാസ്യം ഫോസ്ഫോണേറ്റ് 40 SC (3-4 മില്ലി/ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ മെറ്റലാക്‌സിൽ + മാങ്കോസെബ് (8+64) WP അല്ലെങ്കിൽ ഫോസെറ്റ്‌ലി അൽ 80 WP (2 ഗ്രാം/ലിറ്റർ വെള്ളം) തളിക്കുക.',
       'tmt late blight.mp3', 'tomato late blight mal.m4a'
     ],
     'The above leaf is Tomato Leaf Mold': [
       'Organic Control-Seed treatment with hot water (25 minutes at 122 °F or 50 °C) is recommended to avoid the pathogen on seeds. The fungi Acremonium strictum, Dicyma pulvinata, Trichoderma harzianum or T. viride and Trichothecium roseum are antagonistic to M. fulva and could be used to reduce its spread. In greenhouse trials the growth of M. fulva on tomatoes was inhibited by A. strictum, Trichoderma viride strain 3 and T. roseum by 53, 66 and 84% respectively. In small arms, apple-cider, garlic or milk sprays and vinegar mix can be used to treat the mold.Chemical Control-Always consider an integrated approach with preventive measures together with biological treatments if available. Applications should be made prior to infection when environmental conditions are optimal for the development of the disease. Recommended compounds in field use are chlorothalonil, maneb, mancozeb and copper formulations. For greenhouses, difenoconazole, mandipropamid, cymoxanil, famoxadone and cyprodinil are recommended.', 'ഓർഗാനിക് കൺട്രോൾ-വിത്തുകളിലെ രോഗകാരിയെ ഒഴിവാക്കാൻ ചൂടുവെള്ളം (25 മിനിറ്റ് 122 °F അല്ലെങ്കിൽ 50 °C) ഉപയോഗിച്ചുള്ള വിത്ത് ചികിത്സ ശുപാർശ ചെയ്യുന്നു. അക്രിമോണിയം സ്‌ട്രിക്റ്റം, ഡിസിമ പൾവിനാറ്റ, ട്രൈക്കോഡെർമ ഹാർസിയാനം അല്ലെങ്കിൽ ടി.വിറൈഡ്, ട്രൈക്കോതെസിയം റോസിയം എന്നീ കുമിളുകൾ എം.ഫുൾവയ്‌ക്ക് എതിരായതിനാൽ അതിന്റെ വ്യാപനം കുറയ്ക്കാൻ ഉപയോഗിക്കാം. ഹരിതഗൃഹ പരീക്ഷണങ്ങളിൽ, തക്കാളിയിലെ എം. ഫുൾവയുടെ വളർച്ചയെ എ. സ്‌ട്രിക്റ്റം, ട്രൈക്കോഡെർമ വൈറൈഡ് സ്‌ട്രെയിൻ 3, ടി. റോസിയം എന്നിവ യഥാക്രമം 53, 66, 84% തടഞ്ഞു. ചെറിയ ആയുധങ്ങളിൽ, ആപ്പിൾ-സൈഡർ, വെളുത്തുള്ളി അല്ലെങ്കിൽ പാൽ സ്പ്രേകൾ, വിനാഗിരി മിശ്രിതം എന്നിവ പൂപ്പൽ ചികിത്സിക്കാൻ ഉപയോഗിക്കാം.കെമിക്കൽ കൺട്രോൾ-എപ്പോഴും ലഭ്യമാണെങ്കിൽ ജീവശാസ്ത്രപരമായ ചികിത്സകൾക്കൊപ്പം പ്രതിരോധ നടപടികളുമായി ഒരു സംയോജിത സമീപനം പരിഗണിക്കുക. ക്ലോറോത്തലോനിൽ, മനേബ്, മാങ്കോസെബ്, കോപ്പർ ഫോർമുലേഷനുകൾ എന്നിവയാണ് ഫീൽഡ് ഉപയോഗത്തിൽ ശുപാർശ ചെയ്യുന്ന സംയുക്തങ്ങൾ. ഹരിതഗൃഹങ്ങൾക്ക്, ഡിഫെനോകോണസോൾ, മാൻഡിപ്രോപാമിഡ്, സൈമോക്സാനിൽ, ഫാമോക്സഡോൺ, സൈപ്രോഡിനിൽ എന്നിവ ശുപാർശ ചെയ്യുന്നു.',
       'leaf mold.mp3', 'tomato leaf mold mal.m4a'
     ],
     'The above leaf is Tomato Septoria leaf spot': [
       'It is also helpful to practice a 3- to 4- year rotation with non-Solanaceous crops, if possible, and to control weeds in and around the garden.Staking and mulching can also help reduce infections. Staking increases air circulation and helps to dry the leaves-this reduces favorable conditions for infection. Mulching acts as a barrier between the soil( and the tomato and prevents splashing of spores onto the lower leaves. This is often the primary way the disease gets started each season.The final strategy for managing Septoria leaf spot involves the proper selection, timing, and application of fungicide sprays. Fungicides are very effective and are often necessary to supplement the other management strategies previously outlined. Thorough coverage of all parts of the plant is necessary and the sprays should be applied until run-off. The fungicide label will contain information on dosage rates, pre-harvest interval (PHI), and safety precautions. Among the fungicides registered for homeowner use in Connecticut are chlorothalonil and mancozeb. Organic options include fungicides containing copper, potassium bicarbonate, and QST 713 strain of Bacillus subtilis. In Connecticut, the first sprays are usually needed by mid-July, between flowering and first fruit set.', 'സാധ്യമെങ്കിൽ സോളനേഷ്യസ് അല്ലാത്ത വിളകൾ ഉപയോഗിച്ച് 3 മുതൽ 4 വർഷം വരെ ഭ്രമണം ചെയ്യുന്നതും പൂന്തോട്ടത്തിലും പരിസരത്തും കളകളെ നിയന്ത്രിക്കാനും ഇത് സഹായകരമാണ്.പുതയിടുന്നതും പുതയിടുന്നതും അണുബാധ കുറയ്ക്കാൻ സഹായിക്കും. സ്റ്റാക്കിംഗ് വായുസഞ്ചാരം വർദ്ധിപ്പിക്കുകയും ഇലകൾ ഉണങ്ങാൻ സഹായിക്കുകയും ചെയ്യുന്നു - ഇത് അണുബാധയ്ക്കുള്ള അനുകൂല സാഹചര്യങ്ങൾ കുറയ്ക്കുന്നു. പുതയിടുന്നത് മണ്ണിനും (തക്കാളിക്കും ഇടയിൽ ഒരു തടസ്സമായി പ്രവർത്തിക്കുകയും താഴത്തെ ഇലകളിൽ ബീജങ്ങൾ തെറിക്കുന്നത് തടയുകയും ചെയ്യുന്നു. ഓരോ സീസണിലും രോഗം ആരംഭിക്കുന്നതിനുള്ള പ്രാഥമിക മാർഗമാണിത്.സെപ്റ്റോറിയ ഇലപ്പുള്ളി കൈകാര്യം ചെയ്യുന്നതിനുള്ള അവസാന തന്ത്രം കുമിൾനാശിനി സ്പ്രേകളുടെ ശരിയായ തിരഞ്ഞെടുപ്പ്, സമയം, പ്രയോഗം എന്നിവ ഉൾപ്പെടുന്നു. കുമിൾനാശിനികൾ വളരെ ഫലപ്രദമാണ് കൂടാതെ മുമ്പ് വിവരിച്ച മറ്റ് മാനേജ്മെന്റ് തന്ത്രങ്ങൾക്ക് അനുബന്ധമായി അവ പലപ്പോഴും ആവശ്യമാണ്. ചെടിയുടെ എല്ലാ ഭാഗങ്ങളിലും സമഗ്രമായ കവറേജ് ആവശ്യമാണ്, കൂടാതെ സ്പ്രേകൾ റൺ ഓഫ് വരെ പ്രയോഗിക്കണം. കുമിൾനാശിനി ലേബലിൽ അളവ് നിരക്ക്, വിളവെടുപ്പിന് മുമ്പുള്ള ഇടവേള (PHI), സുരക്ഷാ മുൻകരുതലുകൾ എന്നിവയെക്കുറിച്ചുള്ള വിവരങ്ങൾ അടങ്ങിയിരിക്കും. കണക്റ്റിക്കട്ടിൽ വീട്ടുടമസ്ഥരുടെ ഉപയോഗത്തിനായി രജിസ്റ്റർ ചെയ്ത കുമിൾനാശിനികളിൽ ക്ലോറോത്തലോനിലും മാങ്കോസെബും ഉൾപ്പെടുന്നു. ചെമ്പ്, പൊട്ടാസ്യം ബൈകാർബണേറ്റ്, ബാസിലസ് സബ്‌റ്റിലിസിന്റെ QST 713 സ്‌ട്രെയിൻ എന്നിവ അടങ്ങിയ കുമിൾനാശിനികൾ ഓർഗാനിക് ഓപ്ഷനുകളിൽ ഉൾപ്പെടുന്നു. കണക്റ്റിക്കട്ടിൽ, പൂവിടുന്നതിനും ആദ്യത്തെ കായ്കൾ തുടങ്ങുന്നതിനുമിടയിൽ സാധാരണയായി ജൂലൈ പകുതിയോടെ ആദ്യത്തെ സ്പ്രേകൾ ആവശ്യമാണ്.',
       'tmt septoria.mp3', 'tomato septoria mal.m4a'
     ],
     'The above leaf is Tomato Spider mites Two-spotted spider mite': [
       'Spray the crop with water using a sprayer. Spray rice gruel water on under surface of leaves (1 cup rice gruel+1 cup water). Spray castor oil-soap emulsion or neem oil- garlic emulsion (20 ml neem/caster oil +20 gram garlic paste + 5 gram soap / litre of water).In case of severe infection, spray Wettable Sulphur 80 WP (2 gram/litre of water) or Spiromesifen 240 SC (8 ml/ 10 litre of water) or Fenazaquin 10 EC (2.5 ml/litre of water).', 'സ്പ്രേയർ ഉപയോഗിച്ച് വെള്ളം ഉപയോഗിച്ച് വിളകൾ തളിക്കുക. ഇലകളുടെ ഉപരിതലത്തിൽ അരി അരച്ച വെള്ളം തളിക്കുക (1 കപ്പ് അരിത്തരി+1 കപ്പ് വെള്ളം). ആവണക്കെണ്ണ-സോപ്പ് എമൽഷൻ അല്ലെങ്കിൽ വേപ്പെണ്ണ-വെളുത്തുള്ളി എമൽഷൻ (20 മില്ലി വേപ്പ്/കാസ്റ്റർ ഓയിൽ +20 ഗ്രാം വെളുത്തുള്ളി പേസ്റ്റ് + 5 ഗ്രാം സോപ്പ് / ലിറ്റർ വെള്ളത്തിൽ) തളിക്കുക. കഠിനമായ അണുബാധയുണ്ടെങ്കിൽ, വെറ്റബിൾ സൾഫർ 80 WP (2 ഗ്രാം/ലിറ്റർ) തളിക്കുക. വെള്ളം) അല്ലെങ്കിൽ സ്പിറോമെസിഫെൻ 240 SC (8 ml/10 ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ ഫെനാസാക്വിൻ 10 EC (2.5 ml/ലിറ്റർ വെള്ളം',
       'spider mite.mp3', 'tomato spider mal.m4a'
     ],
     'The above leaf is Tomato Target Spot': [
       'If cultural and sanitation practices alone are not sufficient to manage target spot, fungicides can be used. Its important to follow the instructions and recommendations on the product labels. Here are some fungicides commonly used for target spot control inChlorothalonil, Mancozeb, Copper-based fungicides, Azoxystrobin, Pyraclostrobin,TrifloxystrobinCultural Practices:  Avoid planting tomatoes in the same location year after year. Rotate with non-host crops to reduce disease pressure. Select tomato varieties that are resistant to target spot if O available. Provide adequate spacing between tomato plants to improve air circulation and reduce humidity, which can help prevent disease development. Water Management: Avoid overhead irrigation and water the plants at the base to keep foliage dry, as wet leaves promote disease spread. Remove weeds from the tomato field, as they can serve as hosts for the target spot fungus.Sanitation: Regularly remove and destroy any infected plant debris, including infected leaves and fruits, to reduce the source of inoculum for future infections. Disinfect gardening tools and equipment before using them in different areas of the garden to prevent the spread of the fungus', 'ടാർഗെറ്റ് സ്പോട്ട് കൈകാര്യം ചെയ്യാൻ സാംസ്കാരിക, ശുചിത്വ രീതികൾ മാത്രം പര്യാപ്തമല്ലെങ്കിൽ, കുമിൾനാശിനികൾ ഉപയോഗിക്കാം. ടാർഗെറ്റ് സ്പോട്ട് നിയന്ത്രണത്തിനായി സാധാരണയായി ഉപയോഗിക്കുന്ന ചില കുമിൾനാശിനികൾ ഇതാക്ലോറോത്തലോനിൽ, മാങ്കോസെബ്, ചെമ്പ് അധിഷ്ഠിത കുമിൾനാശിനികൾ, അസോക്സിസ്ട്രോബിൻ, പൈക്ലോസ്ട്രോബിൻ,ട്രൈഫ്ലോക്സിസ്ട്രോബിൻസാംസ്കാരിക രീതികൾ: വർഷാവർഷം ഒരേ സ്ഥലത്ത് തക്കാളി നടുന്നത് ഒഴിവാക്കുക. രോഗസമ്മർദ്ദം കുറയ്ക്കുന്നതിന് ആതിഥേയമല്ലാത്ത വിളകൾ ഉപയോഗിച്ച് തിരിക്കുക. O ലഭ്യമാണെങ്കിൽ, ടാർഗെറ്റ് സ്പോട്ടിനെ പ്രതിരോധിക്കുന്ന തക്കാളി ഇനങ്ങൾ തിരഞ്ഞെടുക്കുക. വായു സഞ്ചാരം മെച്ചപ്പെടുത്തുന്നതിനും ഈർപ്പം കുറയ്ക്കുന്നതിനും തക്കാളി ചെടികൾക്കിടയിൽ മതിയായ അകലം നൽകുക, ഇത് രോഗത്തിന്റെ വികസനം തടയാൻ സഹായിക്കും. ജലപരിപാലനം: നനഞ്ഞ ഇലകൾ രോഗവ്യാപനം പ്രോത്സാഹിപ്പിക്കുന്നതിനാൽ, ഓവർഹെഡ് ജലസേചനം ഒഴിവാക്കുകയും ഇലകൾ വരണ്ടതാക്കാൻ ചുവട്ടിലെ ചെടികൾക്ക് വെള്ളം നൽകുകയും ചെയ്യുക. ടാർഗെറ്റ് സ്പോട്ട് ഫംഗസിന്റെ ആതിഥേയരായി സേവിക്കാൻ കഴിയുന്നതിനാൽ, തക്കാളി വയലിൽ നിന്ന് കളകൾ നീക്കം ചെയ്യുക.ശുചിത്വം: ഭാവിയിലെ അണുബാധകൾക്കുള്ള ഇനോക്കുലത്തിന്റെ ഉറവിടം കുറയ്ക്കുന്നതിന്, രോഗബാധിതമായ ഇലകളും പഴങ്ങളും ഉൾപ്പെടെ, രോഗബാധയുള്ള ചെടികളുടെ അവശിഷ്ടങ്ങൾ പതിവായി നീക്കം ചെയ്യുകയും നശിപ്പിക്കുകയും ചെയ്യുക. പൂന്തോട്ടത്തിനുള്ള ഉപകരണങ്ങളും ഉപകരണങ്ങളും പൂന്തോട്ടത്തിന്റെ വിവിധ ഭാഗങ്ങളിൽ ഉപയോഗിക്കുന്നതിന് മുമ്പ് ഫംഗസ് പടരാതിരിക്കാൻ അണുവിമുക്തമാക്കുക.',
       'target spot.mp3','tomato target spot mal.m4a'  
     ],
    'The above leaf is Tomato Yellow Leaf Curl Virus': [
       'Use disease free seed. Rogue out the infected plants. Alternate or collateral hosts harboring the virus causing this disease should be removed at the time of weeding or earthing up operations to minimize the spread of the disease. Apply neem cake in the planting pits. Spray Dimethoate 30 EC (2 ml/litre of water) to control insect vector.Barrier crops like maize, jowar, bajra are good to protect the crop from TLCV infection. TLCV incidence can be reduced drastically by the use of polythene mulching in the soil just before transplanting of tomato. Polythene sheets of white, blue, grey and black colours are effective.', 'രോഗമില്ലാത്ത വിത്ത് ഉപയോഗിക്കുക. രോഗബാധയുള്ള ചെടികൾ പുറത്തെടുക്കുക. ഈ രോഗത്തിന് കാരണമാകുന്ന വൈറസിനെ സംരക്ഷിക്കുന്ന ഇതര അല്ലെങ്കിൽ കൊളാറ്ററൽ ഹോസ്റ്റുകൾ, രോഗവ്യാപനം കുറയ്ക്കുന്നതിന് കളകൾ നീക്കം ചെയ്യുന്ന സമയത്തോ മണ്ണിടൽ പ്രവർത്തനങ്ങളിലോ നീക്കം ചെയ്യണം. നടീൽ കുഴികളിൽ വേപ്പിൻ പിണ്ണാക്ക് ഇടുക. പ്രാണികളെ നിയന്ത്രിക്കാൻ ഡൈമെത്തോയേറ്റ് 30 ഇസി (2 മില്ലി/ലിറ്റർ വെള്ളം) തളിക്കുക.ചോളം, ജോവർ, ബജ്ര തുടങ്ങിയ തടസ്സവിളകൾ ടിഎൽസിവി അണുബാധയിൽ നിന്ന് വിളയെ സംരക്ഷിക്കാൻ നല്ലതാണ്. തക്കാളി നടുന്നതിന് തൊട്ടുമുമ്പ് മണ്ണിൽ പോളിത്തീൻ പുതയിടുന്നത് വഴി TLCV ബാധ ഗണ്യമായി കുറയ്ക്കാം. വെള്ള, നീല, ചാര, കറുപ്പ് നിറങ്ങളിലുള്ള പോളിത്തീൻ ഷീറ്റുകൾ ഫലപ്രദമാണ്',
       'yellow.mp3', 'tomato yellow curl mal.m4a'
    ],
    'The above leaf is Tomato mosaic virus': [
       'Use disease free seed. Rogue out the infected plants. Apply neem cake in the planting pits. Spray Dimethoate 30 EC (2 ml / litre of water) or Malathion 50 EC (2 ml / litre of water) to control the aphid vectors.', 'രോഗമില്ലാത്ത വിത്ത് ഉപയോഗിക്കുക. രോഗബാധയുള്ള ചെടികൾ പുറത്തെടുക്കുക. നടീൽ കുഴികളിൽ വേപ്പിൻ പിണ്ണാക്ക് ഇടുക. മുഞ്ഞയുടെ രോഗാണുക്കളെ നിയന്ത്രിക്കാൻ ഡൈമെത്തോയേറ്റ് 30 ഇസി (2 മില്ലി / ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ മാലത്തിയോൺ 50 ഇസി (2 മില്ലി / ലിറ്റർ വെള്ളം) തളിക്കുക.',
       'tmt mosaic.mp3', 'cassava.m4a'
    ],
    'The above leaf is bean angular leaf spot': [
       'Use healthy seeds.Adopt 2 year crop rotation in infected field.Spray Carbendazim 50 WP (50 g/100 litre of water) or Mancozeb (250 g/100 litre of water.', 'ആരോഗ്യമുള്ള വിത്തുകൾ ഉപയോഗിക്കുക. രോഗബാധയുള്ള വയലിൽ 2 വർഷത്തെ വിള ഭ്രമണം സ്വീകരിക്കുക. കാർബൻഡാസിം 50 WP (50 ഗ്രാം/100 ലിറ്റർ വെള്ളം) അല്ലെങ്കിൽ മാങ്കോസെബ് (250 ഗ്രാം/100 ലിറ്റർ വെള്ളം) തളിക്കുക',
       'bn angular.mp3', 'bean angular mal.m4a'
    ],
    'The above leaf is bean rust': [
        'Use clean bean seeds originating from non-diseased plants or from certified seed dealers. Using clean seeds prevents spreading of rust disease. Pick affected leaves as soon as symptoms are seen to reduce disease spread. Do not openly carry leaves around, as this spreads the disease. Remove the crop remains from the field and bury them to reduce the spread and the over-season survival of the disease. Fungicides are most effective when used preventively or in the very early stages of the disease. You may use products containing mancozeb which is less toxic than other products (WHO toxicity class U: unlikely hazard). You may use copper oxychloride, or tebuconazole, but these fungicides are toxic to you and consumers (class II: moderately hazardous). Always wear protective clothing during sprays. Always read product labels. Spray maximally twice per season, and stop spraying 35 days before harvest. Nobody should enter sprayed fields during 3 days. If disease symptoms are found on most of the bean plants, spraying is too late.',
        'രോഗമില്ലാത്ത ചെടികളിൽ നിന്നോ സാക്ഷ്യപ്പെടുത്തിയ വിത്ത് ഡീലർമാരിൽ നിന്നോ ഉത്ഭവിക്കുന്ന ശുദ്ധമായ ബീൻസ് വിത്തുകൾ ഉപയോഗിക്കുക. ശുദ്ധമായ വിത്തുകൾ ഉപയോഗിക്കുന്നത് തുരുമ്പ് രോഗം പടരുന്നത് തടയുന്നു. രോഗലക്ഷണങ്ങൾ കണ്ടാലുടൻ രോഗം പടരുന്നത് കുറയ്ക്കാൻ ഇലകൾ പറിച്ചെടുക്കുക. ഇലകൾ പരസ്യമായി കൊണ്ടുനടക്കരുത്, ഇത് രോഗം പരത്തുന്നു.',
        'bean rust.mp3',
        'bean rust mal.m4a'
    ]
    # add remedies for other diseases in both English and Malayalam
}

selected_language = 'English'  # Set the default language


num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model_path = "epoch-8.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
def model_predict(image, model_func, transform):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    output = model_func(image_tensor)
    index = torch.argmax(output)
    pred = classes[index.item()]
    probs, _ = torch.max(F.softmax(output, dim=1), 1)
    if probs < 0.8:
        pred = 'This is not trained yet'
        return pred, probs
    else:
        return pred, probs


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
text_area_style = """
    background-color: black;
    color: white;
    font-family: Arial, sans-serif;
"""
def display_remedies(pred):
    remedy = remedies.get(pred)
    if remedy:
        st.markdown("<p style='color:red;'>Remedy:</p>", unsafe_allow_html=True)
        if selected_language == 'English':
            audio_file = remedy[2]
        else:
            audio_file = remedy[3]
        with open(audio_file, 'rb') as audio:
            st.audio(audio.read(), format='audio/mp3')
        if selected_language == 'English':
            st.text_area("", value=remedy[0], height=200)
        else:
            st.text_area("", value=remedy[1], height=200)

def display_remedies_malayalam(pred):
    remedy = remedies.get(pred)
    if remedy:
        st.markdown("<p style='color:white;'>Remedy (Malayalam):</p>", unsafe_allow_html=True)
        audio_file = remedy[3]
        with open(audio_file, 'rb') as audio:
            st.audio(audio.read(), format='audio/mp3')
        st.text_area("", value=remedy[1], height=200)
# Initialize SessionState
def init_session_state():
    if 'session_state' not in st.session_state:
        st.session_state.session_state = {
            'pred': None,
            'probs': None,
            'selected_language': 'English',
            'language_selected': False
        }

def home_page():
    st.title("Welcome to Dr.Leaf")
    st.write("Instructions:")
    st.write("👉 Take a clear photo of a single leaf.")
    st.write("👉 Ensure that the leaf doesn't have any dust or other unwanted things.")
    st.write("👉 Currently Dr.Leaf can diagonise tomato, beans and cassava.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        st.write("")

        if st.button("Classify"):
            pred, probs = model_predict(image, model, transform)
            st.session_state.session_state['pred'] = pred
            st.session_state.session_state['probs'] = probs.item()
            st.session_state.session_state['language_selected'] = False

    if st.session_state.session_state['pred'] is not None:
        st.markdown(f"<p style='color: white;'>Prediction: {st.session_state.session_state['pred']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: white;'>Probability: {st.session_state.session_state['probs']}</p>", unsafe_allow_html=True)
        if st.session_state.session_state['pred'] != 'This is not trained yet' and st.session_state.session_state['pred'] != 'The above leaf is Cassava Healthy leaf' and st.session_state.session_state['pred'] != ' The above leaf is Tomato healthy' and st.session_state.session_state['pred'] != 'The above leaf is bean healthy':
            selected_language = st.selectbox("Select Language", ['English', 'Malayalam'], index=0, key="language_select")
            st.session_state.session_state['selected_language'] = selected_language

    if st.session_state.session_state['pred'] is not None and not st.session_state.session_state['language_selected']:
        if st.session_state.session_state['selected_language'] == 'Malayalam':
            display_remedies_malayalam(st.session_state.session_state['pred'])
        else:
            display_remedies(st.session_state.session_state['pred'])

def about_page():
    st.title("About Dr.Leaf")
    st.write("Dr.Leaf is a sophisticated and powerful application designed to accurately identify plant diseases based on leaf images. It offers comprehensive remedies for these diseases in both English and Malayalam, catering to a wide range of users. Our primary focus is on individuals actively involved in kitchen gardening, ensuring that they have the necessary tools to maintain the health of their plants.")
    st.write("Dr.Leaf is the result of diligent teamwork by our dedicated team members: Laura, Niyas, Poornima, and Pranav. Their collective efforts have culminated in an application that seamlessly combines advanced technology with practical solutions for plant disease identification and management.")
    st.write("We are committed to providing a user-friendly experience, empowering kitchen gardening enthusiasts to make informed decisions and take appropriate actions to safeguard their plants. Dr.Leaf strives to be a reliable companion for individuals seeking expert guidance in plant disease detection and treatment.")



def contact_page():
    st.title("Contact Dr.Leaf")
    st.write("For any inquiries or support, please contact us at:")
    st.write("- Email ids:")
    st.write("  - laura.saleena@gmail.com")
    st.write("  - niyasmohammad16@gmail.com")
    st.write("  - poornimababus2001@gmail.com")
    st.write("  - pranavrajiv2000@gmail.com")



# Initialize SessionState
def init_session_state():
    if 'session_state' not in st.session_state:
        st.session_state.session_state = {
            'pred': None,
            'probs': None,
            'selected_language': 'English',
            'language_selected': False
        }

def main():
    init_session_state()

    st.set_page_config(page_title="Dr.Leaf", page_icon="logo.png")
    header_container = st.beta_container()

    # Add logo and title to the header
    header_columns = header_container.beta_columns([1, 8])  # Adjust column widths as needed

    with header_columns[0]:
        st.image('logo.png', width=70)

    with header_columns[1]:
        st.title('Dr.Leaf')
    add_bg_from_local("background app2a.jpg")
    # Navigation panel
    st.sidebar.title("Dr.Leaf")
    st.sidebar.image("logo.png")
    nav_selection = st.sidebar.radio("", ["Home", "About", "Contact"])

    if nav_selection == "Home":
        home_page()
    elif nav_selection == "About":
        about_page()
    elif nav_selection == "Contact":
        contact_page()

if __name__ == "__main__":
  main()
