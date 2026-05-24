# Primena neuronskih mreža za predvidjanje prevara u finansijama (credit card fraud)

## 1. Uvod i opis problema

<p align="justify">
U poslednjih nekoliko godina, sa napretkom tehnologije, većina ljudi koristi kreditne kartice za kupovinu i podmirivanje osnovnih životnih potreba, usled čega i broj prevara povezanih sa njima postepeno raste. Danas gotovo sva preduzeća, od malih firmi do velikih korporacija, koriste kreditnu karticu kao način plaćanja. Prevare sa kreditnim karticama dešavaju se u svim organizacijama, uključujući industriju kućnih aparata, automobilsku industriju, banke i slično. Mnogi procesi, poput rudarenja podataka i algoritamskih pristupa mašinskog učenja, primenjivani su za identifikaciju prevara u transakcijama kreditnih kartica, ali nisu dali značajne rezultate. Zbog toga postoji potreba za razvojem efektivnih i efikasnih algoritama koji rade znatno uspešnije. Mi pokušavamo da predvidimo prevarante u korišćenju kreditne kartice pre nego što transakcija uopšte bude odobrena, i to pomoću algoritma veštačkih neuronskih mreža.
</p>

<p align="justify">
Detekcija prevara sa kreditnim karticama odvija se na sledeći način: korisnik unosi neophodne podatke kako bi izvršio transakciju, a transakcija bi trebalo da bude odobrena tek nakon što se proveri da li u njoj ima bilo kakvih nedozvoljenih aktivnosti. Da bi se to postiglo, detalje transakcije prvo prosleđujemo modulu za verifikaciju, gde se ona klasifikuje u kategoriju prevare ili legitimne transakcije. Svaka transakcija koja bude svrstana u kategoriju prevare se odbija. U suprotnom, transakcija se odobrava.
</p>

<p align="justify">
Glavni tehnički izazov u domenu detekcije finansijskih prevara leži u činjenici da su baze podataka ekstremno debalansirane (engl. <i>class imbalance</i>). U realnim uslovima, regularne transakcije čine preko 99% ukupnog saobraćaja, dok se prevare javljaju u svega nekoliko promila ili procenata. Tradicionalni algoritmi mašinskog učenja često podbace u ovakvim scenarijima jer teže da minimizuju ukupnu grešku tako što će jednostavno svaku transakciju proglasiti legitimnom, čime prividno postižu visoku tačnost. Za razliku od drugih algoritama neuronske mreže, zahvaljujući složenim unutrašnjim slojevima i nelinearnim aktivacionim funkcijama, poseduju jedinstvenu sposobnost da uoče suptilne, skrivene obrasce i anomalije koje opisuju prevaru, čak i u uslovima visoke nesrazmere podataka.
</p>

<p align="justify">
Predmet ovog projektnog zadatka jeste implementacija više modela neuronskih mreža sa ciljem binarne klasifikacije transakcija na legitimne i prevarantske. Pored same konstrukcije i obučavanja mreže, poseban fokus biće stavljen na preprocesiranje podataka, rešavanje problema pomenutog debalansa klasa, kao i na hiperparametarsku optimizaciju kroz promenu stopa učenja, aktivacionih funkcija i optimizacionih algoritama. Evaluacija uspešnosti modela biće izvršena korišćenjem naprednih metričkih pokazatelja.
</p>

## 2. Struktura projekta

<p align="justify">
Kod linearnih algoritama s kojima smo se ranije susretali, poput logističke regresije ili  SVM-a, pretpostavka je linearna. Na primer, mogli bismo reći da što je veći iznos transakcije, veća je šansa da je u pitanju prevara.
</p>
<p align="justify">
U realnosti, prevare su mnogo suptilnije. Izvršena transakcija ne mora da bude u iznosu od više stotina eura da bi bila krađa. Možda vršioci ilegalnih aktivnosti naprave desetak brzih transakcija od po 5 eura u roku od svega par minuta. Dakle, anomalija nije u visokoj vrednosti vež u specifičnoj kombinaciji određenih vrednosti pod čudnim uglovima. 
</p>

<p align="justify">
Glavni cilj ovog rada jeste komparativna analiza performansi različitih arhitektura i konfiguracija veštačkih neuronskih mreža nad prikupljenim podacima, i sve to zarad identifikacije optimalnog modela za primenu u realnom bankarskom poslovanju. 
</p>
<p align="justify">
Detekcija finansijskih prevara zahteva balansiranje između dva dijametralno suprotstavljena poslovna cilja: maksimizacije bezbednosti (hvatanja što većeg broja prevara) i očuvanja vrhunskog korisničkog iskustva (izbegavanje lažnih uzbuna i neopravdanog blokiranja kartica).
</p>
<p align="justify">
Kroz razvoj četiri modela, analiziraćemo kako različite inženjerske metode utiču na ovaj kompromis. 
</p>

* Model 1: **Baseline model (skicit-learn MLPClassifier)**
* Model 2: **Osnovni PyTorch MLP model**
* Model 3: **Težinski PyTorch MLP model**
* Model 4: **Regularizovani PyTorch MLP model**

## 3. Podaci  (izvor, struktura, analiza preprocesiranje)

<p align="justify">
Skup podataka koji će biti korišćen nadalje u radu preuzet je sa <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank">sledećeg linka</a>.
Dataset sadrži transakcije napravljene kreditnim karticama u septembru 2013. godine od strane evropskih državljana. Sve transakcije obavljene su u periodu od dva dana, gde je utvrđeno postojanje 492 prevare od sveukupno 284.807 izvršenih transakcija. Ovaj skup podataka je kao što smo u uvodu napomenuli - ekstremno debalansiran. 
Što se tiče atributa, sadrži samo numeričke varijable koje većinski predstavljaju rezultat PCA transformacije radi zaštite privatnosti korisnika. Iz tog razloga ne možemo da donosimo ikakve zaključke o pozadini korisnika. 
</p>

<p align="justify">
U opisu skupa podataka na platformi s koje je preuzet, pominje se da su kolone od V1 do V28 rezultat PCA (Principal Component Analysis) transformacije. Najjednostavnije rečeno, PCA je tehnika koja služi za sažimanje podataka, odnosno smanjenje dimenzionalnosti. Najprostiji primer za to šta tačno radi PCA, mogli bismo da uzmemo iz svakodnevnog života. Ukoliko neko želi da fotografiše drvo, on drvo prebacuje iz tri dimenzije u dve dimenzije (fotografiju). Njegov cilj je da zabeleži to drvo sa najviše detalja, senki i oblika, tako da slika bude što verodostojnija i pored toga što je jedna dimenzija nepovratno izgubljena. Ukoliko se odabere loš ugao, neko ko bude posmatrao fotografiju, teže će zaključiti koja je biljka u pitanju. Upravo tako, PCA radi sa podacima – uzima ogroman broj početnih kolona i pronalazi najbolji matematički ugao pod kojim može da ih sabije u manji broj kolona, a da se pritom zadrži suština i ključni šabloni.
</p>

<p align="justify">
U našem skupu podataka, banka je u početku imala vrlo jasne i detaljne podatke o svakoj transakciji: tačnu geografsku lokaciju, ime prodavnice, kategoriju kupljene robe, IP adresu uređaja, istoriju kupovine tog konkretnog klijenta i slično. Pošto su to strogo poverljivi podaci koji narušavaju privatnost korisnika, banka je primenila PCA algoritam. PCA je uzeo sve te stotine tajnih informacija, matematički ih promešao i sabio u 28 anonimnih kolona (od V1 do V28).
</p>

<p align="justify">
Jedine varijable koje su ostale u svom originalnom obliku su <b>Amount</b> i <b>Time</b>. Amount predstavlja iznos transakcije i njena vrednost znatno varira za svaku instancu (može da iznosi od par do više stotina eura). Time sadrži sekunde protekle između svake transakcije i prve transakcije u skupu. Varijabla <b>Class</b> predstavlja ciljnu promenjivu i sadrži vrednosti 1 (za prevaru) i 0 (u suprotnom).
</p>

<p align="justify">
Radi bolje preglednosti, u tabeli ispod je prikazana konačna struktura atributa skupa podataka koji će biti korišćeni za obučavanje neuronske mreže:
</p>

<div align="center">
  
| Naziv atributa | Tip podatka | Opis atributa |
| :--- | :--- | :--- |
| **V1 - V28** | Numerički (Float) | Kompresovane anonimne karakteristike dobijene PCA transformacijom |
| **Time** | Numerički (Float) | Protekle sekunde od prve transakcije u bazi podataka |
| **Amount** | Numerički (Float) | Novčani iznos transakcije (izražen u eurima) |
| **Class** | Binarni (0 ili 1) | **Ciljna promenljiva** (0 = Regularna transakcija, 1 = Prevara) |

</div>

## 4. Arhitektura modela

<div align="center">
<img width="800" height="340" alt="arhitektura_neuronske_mreze" src="https://github.com/user-attachments/assets/0d6c8399-e40f-4920-a3fd-b401796ceafc" />
  <p align="justify">
    <br>
    <b>Slika 1:</b> Detaljan šematski prikaz arhitekture implementiranog višeslojnog perceptrona (MLP). Na šemi je prikazan skraćeni vizuelni pregled slojeva sa tačnim brojem neurona (30 -> 64 -> 32 -> 1).
  </p>
</div>

<p align="justify">Kako bi se osigurala doslednost eksperimenta i omogućilo adekvatno upoređivanje performansi, svi modeli obučavani u okviru ovog istraživanja dele identičnu osnovnu arhitekturu. Kao što je prikazano na dijagramu, svaka neuronska mreža je dizajnirana sa ulaznim slojem od 30 neurona, dva skrivena sloja koja sadrže 64 i 32 neurona respektivno, kao i izlaznim slojem sa jednim neuronom zaduženim za binarnu klasifikaciju.</p>

## 5. Trening

### Model 1 - Baseline model (scikit-learn MLPClassifier)
<p align="justify">
Model 1 je kreiran upotrebom klase <b>MLPClassifier</b> unutar <i>scikit-learn</i> biblioteke sa parametrima <code>hidden_layer_sizes=(64, 32)</code>, <code>max_iter=1000</code> i fiksiranim <code>random_state</code>. Za optimizaciju je upotrebljen Adam algoritam, a obučavanje je zaustavljeno dinamički - mreža je konvergirala u tačno <b>28. iteraciji</b> aktiviranjem mehanizma ranog zaustavljanja (<i>early stopping</i>).
</p>
<p align="justify">
Na grafikonu krive učenja uočava se izrazita nestabilnost i prisustvo velikih oscilacija, uzrokovanih ekstremalnim disbalansom klasa i suboptimalnom fiksnom stopom učenja. Uprkos tome, finalna vrednost loss-a iznosi 0.042.
</p>

### Model 2 - Osnovni PyTorch MLP
<p align="justify">
Prelazak na PyTorch okruženje donosi potpunu kontrolu nad trening petljom. Model je obučavan kroz <b>20 epoha</b> sa mini-serijama veličine 1024, korišćenjem <code>BCEWithLogitsLoss</code> funkcije greške i Adam optimizatora sa stopom učenja 0.001. Kriva učenja pokazuje monoton pad od 0.08 do 0.0017, bez naglih oscilacija - što ukazuje na stabilnu konvergenciju.
</p>

### Model 3 - Težinski PyTorch MLP
<p align="justify">
Identična arhitektura i trening procedura kao Model 2, uz jednu ključnu modifikaciju: funkciji greške <code>BCEWithLogitsLoss</code> prosleđen je parametar <code>pos_weight</code>, izračunat kao odnos broja regularnih transakcija i broja prevara u trening skupu (~578). Ovaj koeficijent primorava optimizator da višestruko strože kažnjava model za svaku propuštenu prevaru, čime se kompenzuje ekstreman disbalans klasa.
</p>

### Model 4 - Regularizovani PyTorch MLP
<p align="justify">
Model 4 zadržava težinsku funkciju greške iz Modela 3 i dodaje dve tehnike regularizacije: <b>Dropout</b> slojeve sa verovatnoćom p=0.2 iza oba skrivena sloja, i <b>L2 regularizaciju</b> (<code>weight_decay=1e-4</code>) unutar Adam optimizatora. Cilj je bio smanjiti preprilagođavanje (<i>overfitting</i>) koje je uzrokovalo veliki broj lažnih uzbuna kod Modela 3.
</p>

## 6. Analiza osetljivosti i hiperparametarska optimizacija

### Threshold analiza

<p align="justify">
Threshold analiza sprovedena je kao pokušaj da se poboljšaju performanse Modela 3 i Modela 4, koji su uprkos visokom recallu pokazali izrazito nisku preciznost pri standardnom pragu odlučivanja od 0.5. Ideja je bila da podizanjem praga model postane selektivniji u proglašavanju transakcija prevarama, čime bi se smanjio broj lažnih uzbuna uz zadržavanje visokog odziva.
</p>
<p align="justify">
Međutim, rezultati analize su pokazali da ni pri optimalnom pragu od 0.85 preciznost Modela 3 ne prelazi 0.15, dok F1-score ostaje na svega 0.256. Ovo ukazuje da problem nije u pragu odlučivanja, već u samoj prirodi modela - uvođenje <code>pos_weight</code> parametra toliko je pomerilo granicu odlučivanja ka manjinskoj klasi da model strukturalno generiše veliki broj lažnih uzbuna, bez obzira na naknadna podešavanja. Threshold tuning dakle nije dao željene rezultate, što dodatno učvršćuje zaključak da je <b>Model 2 optimalan izbor</b>.
</p>

### Analiza osetljivosti

<p align="justify">
Analiza osetljivosti sprovedena je za Model 2 (Baseline Pytorch), kao model koji je pokazao najbolje ukupne performanse, kako bi se stekao uvid u to koja obeležja najviše utiču na njegove odluke.</p>

<p align="justify">Model 2 pokazuje raspodelu uticaja među top obeležjima: V6, V14 i Amount nalaze se na vrhu liste sa vrednostima apsolutnog gradijenta između 0.0008 i 0.001, dok ostatak top 10 (V27, V4, V28, V7, V21, V16, V2) blago opada u opsegu 0.0005–0.00065.
</p>

<p align="justify">
Posebno je značajna visoka pozicija varijable Amount (3. mesto), jedne od dve neskrivene i direktno interpretabilne varijable u datasetu. Njen uticaj ima jasno poslovno opravdanje — prevare često uključuju transakcije neobičnih iznosa, pa je logično da model ovaj signal koristi pri donošenju odluke. Model koji svoju odluku gradi na većem broju relevantnih obeležja, umesto da se oslanja na jedan dominantan signal, generalno je robusniji na šum u podacima i bolje se generalizuje na neviđenim primerima. Ovaj nalaz konzistentan je sa svim prethodno izmerenim metrikama i dodatno učvršćuje preporuku da Model 2 predstavlja optimalno rešenje za detekciju prevara u ovom kontekstu.</p>

## 7. Rezultati evaluacije

<div align="center">
  
| Model | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Model 1 — Sklearn Baseline | 0.60 | 0.80 | 0.69 |
| Model 2 — PyTorch Base | 0.82 | 0.82 | 0.82 |
| Model 3 — Težinski | 0.06 | 0.91 | 0.11 |
| Model 4 — Regularizovani | 0.05 | 0.92 | 0.10 |

</div>

### Model 1 - Baseline model (scikit-learn MLPClassifier)
<p align="justify">Model je uspešno identifikovao 78 stvarnih prevara (True Positive) i tačno klasifikovao 56.813 regularnih transakcija (True Negative), uz 20 propuštenih prevara (False Negative) i 51 lažnu uzbunu (False Positive). F1-score za klasu prevara iznosi <b>0.69</b> i služi kao polazna tačka za poređenje.</p>

### Model 2 - Osnovni PyTorch MLP
<p align="justify">Sa svega 17 lažnih uzbuna i 18 propuštenih prevara, Model 2 ostvaruje uniformnih <b>0.82</b> za preciznost, recall i F1-score što predstavlja značajan napredak u odnosu na Baseline. Makro prosek od 0.91 potvrđuje da model podjednako dobro funkcioniše na obe klase uprkos disbalansu.</p>

### Model 3 - Težinski PyTorch MLP
<p align="justify">Uvođenjem <code>pos_weight</code> recall skače na <b>0.91</b> (samo 9 propuštenih prevara), ali preciznost pada na svega <b>0.06</b> uz 1.418 lažnih uzbuna. F1-score iznosi 0.11, što model čini nepraktičnim za realnu primenu.</p>

### Model 4 - Regularizovani PyTorch MLP
<p align="justify">Dropout i L2 regularizacija nisu poboljšali situaciju - recall raste na <b>0.92</b> (8 propuštenih prevara), ali preciznost ostaje na <b>0.05</b> uz 1.579 lažnih uzbuna. F1-score od 0.10 je najniži od svih modela, što potvrđuje da regularizacija nije adekvatno rešenje za problem ekstremalnog disbalansa klasa u ovom kontekstu.</p>

## 8. Diskusija

<p align="justify"> Rezultati eksperimenta otkrivaju jasnu dinamiku kompromisa između preciznosti i odziva kroz četiri različite konfiguracije modela, i ukazuju na nekoliko važnih zaključaka o prirodi problema detekcije finansijskih prevara. </p> <p align="justify"> <b>Model 2 se izdvaja kao statistički najizbalansiranije i praktično najupotrebljivije rešenje.</b> Uniformne vrednosti od 0.82 za sve tri ključne metrike nisu slučajnost - one su direktna posledica toga što model uči opšte obrasce prevare bez veštačkog pomeranja granice odlučivanja. Analiza osetljivosti dodatno potvrđuje ovu ocenu: Model 2 gradi svoju odluku na ravnomernoj raspodeli uticaja među više relevantnih obeležja (V6, V14, Amount, V27), umesto da se oslanja na jedan dominantan signal. Ovakva arhitektura odlučivanja je robusnija na šum i bolje se generalizuje na neviđenim primerima. </p> <p align="justify"> <b>Modeli 3 i 4 demonstriraju tipičnu zamku agresivnog balansiranja klasa.</b> Uvođenjem <code>pos_weight</code> koeficijenta od ~578, funkcija greške postaje toliko opsednuta manjinskom klasom da model praktično svaku transakciju proglašava sumnjivom. Threshold analiza, sprovedena u opsegu 0.1–0.9, pokazala je da ni pri optimalnom pragu od 0.85 preciznost Modela 3 ne prelazi 0.15, što dokazuje da problem nije u izboru praga već u fundamentalnoj konfiguraciji modela. Regularizacija (Dropout + L2) u Modelu 4 nije uspela da ispravi ovaj disbalans, jer je problem strukturalnog, a ne generalizacionog karaktera. </p> <p align="justify"> <b>Važno ograničenje projekta</b> tiče se interpretabilnosti rezultata: varijable V1–V28 su anonimizovane PCA transformacijom, zbog čega nije moguće dati pun poslovni kontekst za nalaze analize osetljivosti. Jedini direktno interpretabilan signal - varijabla Amount - pojavljuje se na trećem mestu po uticaju u Modelu 2, što je konzistentno sa opštepoznatim obrascima finansijskih prevara. </p> <p align="justify"> <b>Moguća unapređenja</b> koja nisu bila predmet ovog rada uključuju: primenu SMOTE tehnike za sintetičko generisanje uzoraka manjinske klase, uvođenje validacionog skupa za praćenje generalizacije tokom treninga, kao i ispitivanje naprednih arhitektura poput LSTM mreža koje bi mogle da iskoriste temporalnu strukturu podataka (varijabla Time). </p>

## 9. Zaključak

<p align="justify">
U ovom radu implementirane su i evaluirane četiri konfiguracije višeslojnih perceptrona za detekciju prevara u transakcijama kreditnih kartica, uz fokus na rešavanje problema ekstremnog disbalansa klasa (svega ~0.17% prevara u skupu).
</p>
<p align="justify">
<b>Model 2 (osnovni PyTorch MLP)</b> pokazao se kao optimalno rešenje sa uniformnim F1-score-om od 0.82 za klasu prevara, minimalnim brojem lažnih uzbuna (17) i stabilnom krivom učenja. Analiza osetljivosti potvrđuje da model donosi odluke na osnovu više relevantnih signala, što ga čini otpornim i pouzdanim. Threshold analiza sprovedena nad Modelima 3 i 4 pokazala je da ni podešavanje praga odlučivanja ne može kompenzovati strukturalni problem agresivnog balansiranja klasa — čime je superiornost Modela 2 dodatno učvršćena.
</p>
<p align="justify">
Ključna lekcija ovog projekta jeste da u domenima sa ekstremnim disbalansom klasa, sofisticiranija tehnika ne znači nužno bolji rezultat. Model koji minimalistički i stabilno uči granice između klasa — bez agresivnih intervencija u funkciji greške — pokazao se superiornijim od modela sa eksplicitnim mehanizmima balansiranja. Preciznost od 0.82 i recall od 0.82 čine Model 2 direktno primenljivim u realnom bankarskom okruženju, gde je podjednako važno uhvatiti prevaru i ne blokirati legitimnog korisnika.
</p>

## Autori

* Anja Perović 2022/0174
* Una Ilić 2022/0291
