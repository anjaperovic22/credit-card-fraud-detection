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

## 5. Model 1 - Baseline model (skicit-learn MLPClassifier)

<p align="justify">
U svetu mašinskog učenja, <b>MLPClassifier</b> (Multi-layer Perceptron Classifier) iz biblioteke <i>scikit-learn</i> predstavlja pristupačnu i moćnu polaznu tačku za rešavanje problema klasifikacije pomoću neuronskih mreža. Ovaj model nudi implementaciju višeslojnog perceptrona i predstavlja idealan izbor za izgradnju robustnog baseline modela, pre nego što se pređe na složenije deep learning okvire.
</p>

### Pravljenje modela

<p align="justify">
Model 1 je kreiran upotrebom klase <b>MLPClassifier</b> unutar <i>scikit-learn</i> biblioteke. Struktura mreže i parametri optimizacije definišu se implicitno prilikom same inicijalizacije objekta. Skriveni slojevi su deklarisani pomoću <code>hidden_layer_sizes=(64, 32)</code>. Scikit-Learn automatski mapira ulazni sloj na osnovu dimenzionalnosti prosleđene matrice podataka ($X$) i povezuje ga sa prvim skrivenim slojem od 64 neurona, koji se dalje transformiše u sloj od 32 neurona. Parametar <code>max_iter=1000</code> postavlja gornju granicu za broj iteracija (epoha) optimizatora. Ova vrednost je svesno podignuta na 1000 kako bi se algoritmu pružio dovoljan prostor da bezbedno konvergira do optimalnog lokalnog minimuma, sprečavajući prerani prekid obučavanja usled dostizanja podrazumevanog hardverskog limita. Parametar <code>random_state</code> je vezan za globalnu konstantu projekta, čime je osigurano da se početna pseudo-slučajna inicijalizacija težina (Glorot/Xavier) ponavlja na identičan način prilikom svakog pokretanja ćelije.
</p>


### Trening

<p align="justify">
Za iterativno ažuriranje težina i minimizaciju Log-Loss funkcije greške, upotrebljen je Adam (Adaptive Moment Estimation) optimizator. Ovaj algoritam je implementiran zbog svoje visoke računske efikasnosti i sposobnosti da adaptivno prilagođava stopu učenja za svaki parametar individualno.
</p>
<p align="justify">
Za razliku od modela sa eksplicitno definisanim brojem epoha, trening Modela 1 zasnovan je na dinamičkom mehanizmu tolerancije zaustavljanja. Postavljen na maksimalno 1000 iteracija, algoritam kontinuirano evaluira pad funkcije greške. Obučavanje se automatski zaustavlja kada se detektuje da dalja optimizacija ne donosi statistički relevantno smanjenje gubitka, čime se efikasno limitira potencijalno preprilagođavanje (<i>overfitting</i>). U okviru sprovedenog eksperimenta, mreža je uspešno pronašla optimalni lokalni minimum i konvergirala u tačno <b>28</b>. iteraciji.
</p>
<div align="center">
<img width="700" height="471" alt="grafik1" src="https://github.com/user-attachments/assets/cbb1446a-1630-43d8-9e70-164bbb3ef9bc" />
</div>

<p align="justify">
Na osnovu prikazanog grafika promene vrednosti funkcije greške tokom treninga, uočava se izrazita nestabilnost i prisustvo velikih oscilacija iz epohe u epohu. U stabilnim uslovima obučavanja, kriva greške bi trebalo da ima linearan ili gladak eksponencijalni pad, dok ovde vidimo nagle skokove vrednosti (npr. oko 4, 13, 20. i 24. iteracije).
</p>

<p align="justify">
Ovakvo ponašanje MLPClassifier modela direktno je uzrokovano specifičnostima našeg skupa podataka i ograničenjima podrazumevanih algoritama optimizacije:
</p>

*    <p align="justify"><b>Ekstremni debalans klasa</b>: Tokom stohastičkog ili mini-batch gradijentnog spusta, algoritam deli podatke u manje pakete. S obzirom na to da prevare čine manje od 1% skupa, model u većini paketa vidi samo regularne transakcije i uspešno minimizuje grešku. Međutim, kada model naiđe na paket koji sadrži nekoliko prevara, on pravi drastičnu grešku u predikciji jer te primere retko viđa. To dovodi do naglog ažuriranja težina i skoka celokupne funkcije greške u toj iteraciji.</p>
*    <p align="justify"><b>Suboptimalna stopa učenja</b>: Visoka fiksna stopa učenja uzrokuje da model previše agresivno menja svoje parametre nakon svake greške. Umesto da lagano konvergira ka globalnom minimumu, model "preskače" optimalne vrednosti težina i luta kroz prostor performansi.</p>

### Kreiranje predikcija

<p align="justify">
Kod <i>skicit-learn</i> modela, proces kreiranja predikcija je visokog nivoa apstrakcije i realizovan je putem ugrađene predict() metode. Model interno izračunava verovatnoće pripadnosti klasama, a zatim automatski primenjuje podrazumevani prag odlučivanja (decision threshold) od 0.5. Sve transakcije sa verovatnoćom jednakom ili većom od ovog praga klasifikuju se kao prevare (klasa 1), dok se ostale označavaju kao regularne (klasa 0).
</p>

<code>y_pred_1 = model_1_sklearn_mlp.predict(X_test)
y_proba_1 = model_1_sklearn_mlp.predict_proba(X_test)
</code>

### Evaluacija

### -> Matrica konfuzije

<div align="center">
  <img width="590" height="490" alt="cm1" src="https://github.com/user-attachments/assets/173289e9-0a42-4191-99aa-b6aedd2b85ef" />
</div>

<p align="justify">Model je uspešno identifikovao 78 stvarnih prevara (<b>True Positive</b>) i tačno klasifikovao 56.813 regularnih transakcija (<b>True Negative</b>). S obzirom na to da je ovo inicijalni model pokrenut bez naprednih tehnika balansiranja, sposobnost mreže da locira većinu prevara predstavlja solidnu polaznu osnovu.</p>

<p align="justify">Pored solidnog starta, model pokazuje dva ključna nedostatka. Prvi je prisustvo 20 lažno negativnih rezultata (<b>False Negative</b>) – to su stvarne prevare koje je model propustio i označio kao bezopasne, što u realnom bankarskom sistemu generiše direktne finansijske gubitke. Drugi nedostatak je 51 lažno pozitivna predikcija (<b>False Positive</b>), odnosno situacije u kojima su regularni korisnici greškom blokirani, što narušava korisničko iskustvo.</p>

### -> Klasifikacioni izveštaj za Baseline Model

<div align="center">

| Klasa / Metrika | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Regularna (0)** | 1.00 | 1.00 | 1.00 | 56864 |
| **Prevara (1)** | 0.60 | 0.80 | 0.69 | 98 |
| **Accuracy** | — | — | 1.00 | 56962 |
| **Macro Avg** | 0.80 | 0.90 | 0.84 | 56962 |
| **Weighted Avg** | 1.00 | 1.00 | 1.00 | 56962 |

</div>

<p align="justify"> Ukupna nominalna tačnost modela (<b>Accuracy</b>) iznosi maksimalnih 1.00. Međutim, ova vrednost je visoka isključivo zbog ekstremnog prisustva većinske klase (56.864 uzoraka). Ovo je odličan primer zašto je ukupna tačnost potpuno nepouzdana i varljiva metrika za evaluaciju modela u uslovima visokog debalansa podataka. </p>

<p align="justify">Fokusiranjem na manjinsku klasu (98 uzoraka), uočavamo da model ostvaruje odziv (<b>Recall</b>) od 0.80, što znači da uspešno detektuje 80% stvarnih prevara. Sa druge strane, preciznost (<b>Precision</b>) iznosi 0.60, što ukazuje na to da su 40% transakcija koje je model označio kao sumnjive zapravo bile regularne.</p>

<p align="justify">Harmonijska sredina ovih performansi izražena je kroz <b>F1-score</b> koji za klasu prevara iznosi 0.69. Ovaj broj predstavlja zvaničnu ocenu našeg trenutnog modela i služi kao polazna tačka. Glavni cilj u nastavku projekta biće optimizacija mreže kroz PyTorch okruženje kako bi se ovaj F1-score značajno unapredio.</p>

## 6. Model 2 - Baseline Pytorch model

<p align="justify">Model 2 predstavlja ključnu tranziciju sa tradicionalnih biblioteka visokog nivoa apstrakcije (Scikit-Learn) na namenski okvir za duboko učenje – PyTorch. Primarni cilj ovog modela jeste uspostavljanje izvornog dubokog baseline-a na nivou neuronskih mreža, kako bi se evaluiralo kako se čista arhitektura ponaša pod uticajem ekstremnog disbalansa klasa, pre nego što se primene specifične tehnike otežavanja ili regularizacije.</p>

<p align="justify">Najpre smo izvršile transformaciju train i test skupova u tenzore da bismo mogle da radimo sa PyTorch bibliotekom.</p>

###  Pravljenje modela

<p align="justify">Arhitektura je realizovana kroz prilagođenu klasu FraudDetectionMLP koja nasleđuje bazni modul nn.Module, što omogućava potpunu kontrolu nad matematičkim operacijama u slojevima i protokom gradijenata. Konstruktor klase (<code>__init__</code>) eksplicitno definiše slojeve i njihove transformacije:</p>

<p align="justify"><b>Prvi linearni sloj (nn.Linear(30, 64))</b>: Prihvata ulazni vektor od 30 obeležja (28 PCA komponenti uz skalirano vreme i iznos) i projektuje ga u prostor od 64 dimenzije. Nad izlazom ovog sloja primenjuje se nelinearna aktivaciona funkcija ReLU (nn.ReLU), koja omogućava mreži da uči složene, nelinearne odnose među finansijskim varijablama.</p>
<p align="justify"><b>Drugi linearni sloj (nn.Linear(64, 32))</b>: Uzima apstraktne karakteristike izvučene iz prethodnog sloja i dodatno ih sažima na 32 dimenzije, ponovo koristeći ReLU nelinearnost za očuvanje kompleksnosti reprezentacije.
<p align="justify"><b>Konačni izlazni sloj (nn.Linear(32, 1))</b>: Transformiše 32 obeležja u jedan linearni izlaz (tzv. logit). Važna inženjerska specifičnost ovog modela jeste da izlazni sloj ne sadrži ugrađenu Sigmoid funkciju unutar same arhitekture; sirove vrednosti se direktno prosleđuju, što je optimizovano rešenje za numerički stabilno računanje složenih funkcija greške.

<p align="justify">Metoda <code>forward</code> precizno definiše algoritam prolaza podataka unapred (forward pass), usmeravajući ulazni tenzor kroz definisane linearne transformacije i aktivacione funkcije do konačnog izlaza. Neposredno pre samog kreiranja objekta model_2_pytorch_base, obavezno se poziva funkcija <code>set_seed(RAND_STATE)</code>. Ovaj korak je kritičan jer PyTorch generator nasumičnih brojeva preuzima kontrolu nad inicijalizacijom težina unutar nn.Linear modula. Pozivanjem ove funkcije tačno iznad instance modela, garantuje se da će mreža uvek startovati sa identičnim težinama, čineći ponašanje ovog dubokog modela potpuno ponovljivim.</p>

### Trening

<p align="justify">Za razliku od <i>scikit-learn</i> okruženja gde je proces obučavanja potpuno apstrahovan unutar jedne metode, <i>PyTorch</i> zahteva eksplicitnu definiciju petlje za treniranje (<i>Training Loop</i>). Time se ostvaruje potpuna inženjerska kontrola nad računanjem gradijenata, propagacijom greške i ažuriranjem težina modela.</p>

<p align="justify"><b>Funkcija greške i optimizator:</b></p>

* <p align="justify"><b>Kriterijum (Loss Function):</b> Upotrebljena je <code>BCEWithLogitsLoss</code> funkcija (Binarna unakrsna entropija sa logitima). Ključna prednost ove funkcije je ta što interno kombinuje <code>Sigmoid</code> sloj i standardni <code>BCELoss</code> u jednu matematičku celinu. Ovaj pristup je numerički drastično stabilniji od odvojenog propuštanja vrednosti kroz aktivacionu funkciju unutar arhitekture, čime se sprečavaju problemi sa iščezavajućim gradijentima (<i>vanishing gradients</i>).</p>
* <p align="justify"><b>Optimizator:</b> Optimizacija svih parametara mreže poverena je <b>Adam</b> algoritmu, uz fiksiranu stopu učenja (<i>Learning Rate</i>) od 0.001, što osigurava brzu i stabilnu konvergenciju.</p>

<p align="justify"><b>Iterativni proces i propagacija kroz serije (Epochs & Batches):</b></p>

<p align="justify">Obučavanje je sprovedeno kroz 20 fiksnih epoha. Unutar svake epohe, mreža iterira kroz <i>DataLoader</i> objekat koji joj doprema podatke u mini-serijama (<i>batches</i>) veličine 1024 transakcije. Za svaku pojedinačnu seriju strogo se poštuje sledeći redosled operacija:</p>

* <p align="justify"><b>Resetovanje gradijenata (<code>optimizer.zero_grad()</code>):</b> PyTorch po podrazumevanim podešavanjima akumulira gradijente kroz iteracije. Pre svakog novog koraka neophodno je programski očistiti stare gradijente kako ne bi došlo do neželjenog mešanja matematičkih vektora iz prethodnih serija.</p>
* <p align="justify"><b>Prolaz unapred (<i>Forward pass</i>):</b> Model prima trenutnu seriju atributa i izračunava sirove izlazne vrednosti (<i>logits</i>).</p>
* <p align="justify"><b>Računanje greške (<i>Loss computation</i>):</b> Funkcija greške upoređuje izlaze modela sa stvarnim klasama i precizno kvantifikuje odstupanje, odnosno gubitak.</p>
* <p align="justify"><b>Prolaz unazad (<i>Backward pass</i>):</b> Primenom lanca izvoda (<i>Backpropagation</i>), izračunavaju se gradijenti greške u odnosu na svaku pojedinačnu težinu unutar slojeva neuronske mreže.</p>
* <p align="justify"><b>Ažuriranje težina (<i>Optimization step</i>):</b> Optimizator prilagođava težine neurona u smeru negativnog gradijenta, čime se minimizuje ukupna greška za sledeću iteraciju.</p>

<p align="justify">Sve vrednosti grešaka na nivou serija se akumuliraju i usrednjavaju, čime se dobija prosečan <i>Loss</i> za celu epohu. Ove vrednosti se trajno čuvaju u listi <code>pytorch_loss_history</code> radi naknadne vizuelizacije krive učenja i provere stabilnosti konvergencije sistema.</p>

<div align="center">
<img width="717" height="471" alt="grafik2" src="https://github.com/user-attachments/assets/c0a029d4-889a-4459-ac7b-746544434c50" />
</div>

<p align="justify">Vrednosti gubitka tokom epoha prikazane su na grafikonu iznad. Gubitak je konstantno opadao od početnih 0.08 do 0.0017 u 20. epohi, što ukazuje da je model uspešno naučio obrasce u podacima. Odsustvo naglih skokova i monotoni pad sugerišu stabilnu konvergenciju. Nema znakova preteranog prilagođavanja (overfitting) na samom trening skupu, ali je neophodno proveriti i performanse na test skupu.</p>

### Evaluacija

Nakon završenog procesa obučavanja, pokrenut je postupak evaluacije modela nad testnim skupom podataka kroz sledeće faze:

<p align="justify"><b>1. Prebacivanje modela u režim evaluacije</b>: Model smo eksplicitno prebacili u režim rada za testiranje. Ovo privremeno deaktivira specifične trening slojeve, obezbeđujući konzistentnost predikcija.</p>
<p align="justify"><b>2. Isključivanje računanja gradijenata</b>: Pošto se nad testnim podacima ne vrši obučavanje niti ažuriranje težina, ovaj blok sprečava PyTorch da gradi računarski graf i pamti gradijente. Time se drastično smanjuje potrošnja memorije i ubrzava izvršavanje koda.</p>

<p align="justify"><b>3. Konverzija izlaza u binarne klase (Sigmoid + Threshold)</b>: Budući da model u poslednjem sloju generiše sirove vrednosti (engl. <i>logits</i>), funkcija <code>torch.sigmoid(outputs)</code> kompresuje izlaze u opseg [0, 1], transformišući ih u stvarne verovatnoće prevare.</p>

   * <p align="justify">Primenom praga odlučivanja od 0.5, sve transakcije sa većom verovatnoćom automatski dobijaju klasu 1 (Prevara), dok ostale dobijaju klasu 0 (Regularna transakcija).</p>

<p align="justify">Konačni rezultati se ravnaju i prebacuju iz PyTorch tenzora u standardne NumPy nizove radi dalje metričke evaluacije.</p>

### -> Matrica konfuzije

<div align="center">
<img width="590" height="490" alt="cm2" src="https://github.com/user-attachments/assets/8153a576-b7f5-42f6-94d4-c2087b69f793" />
</div>

<p align="justify">Inicijalni PyTorch model uspešno je identifikovao 80 stvarnih prevara (<b>True Positive</b>) i tačno klasifikovao 56.847 regularne transakcije (<b>True Negative</b>). Činjenica da je mreža već u svom osnovnom obliku, bez dodatnog otežavanja klasa, uspela da locira ~80% anomalija pokazuje da prelazak na dinamičko PyTorch okruženje i Adam optimizator daje izuzetno agresivnu granicu odlučivanja.</p>

<p align="justify">
Sa svega 17 lažnih uzbuna (<b>False Positive</b>) i 18 propuštenih prevara (<b> False Negative</b>) vidljivo je značajno poboljšanje u odnosu na <i>skicit-learn</i> model.
</p>

### -> Klasifikacioni izveštaj za Baseline Model

<div align="center">

| Klasa / Metrika | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Regularna (0)** | 1.00 | 1.00 | 1.00 | 56864 |
| **Prevara (1)** | 0.82 | 0.82 | 0.82 | 98 |
| **Accuracy** | — | — | 1.00 | 56962 |
| **Macro Avg** | 0.91 | 0.91 | 0.91 | 56962 |
| **Weighted Avg** | 1.00 | 1.00 | 1.00 | 56962 |

</div>

<p align="justify">
Nominalna ukupna tačnost modela (<b>Accuracy</b>) iznosi visokih 1.00 (100%).
</p>

<p align="justify">
Za klasu prevara (1), koja je ključna za poslovanje, model ostvaruje:

*   **Preciznost** = 0.82 – od svih transakcija koje je model označio kao sumnjive, 82% su zaista prevare, što znači da je samo 18% lažnih uzbuna.
*   **Odziv** = 0.82 – model uspešno detektuje 82% svih stvarnih prevara, što je solidan nivo osetljivosti.
*   **F1-score** = 0.82 – harmonijska sredina preciznosti i odziva potvrđuje dobru izbalansiranost ove dve metrike.

**Makro prosek** (macro avg) od 0.91 za preciznost, odziv i F1 pokazuje da model podjednako dobro radi na obe klase, uprkos debalansu.
</p>

<p align="justify">
Osnovni PyTorch model predstavlja značajan napredak u odnosu na inicijalni scikit-learn model. Sa F1-score-om od 0.82 za klasu prevara, ovaj model nudi solidnu osnovu za dalje finom podešavanje. Niska stopa lažnih uzbuna (preciznost 0.82) čini ga pogodnim za realnu primenu, dok bi se daljim povećanjem težine klase prevara mogao dodatno poboljšati odziv, uz potencijalni blagi pad preciznosti.
</p>
