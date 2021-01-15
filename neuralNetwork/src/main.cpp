#include <vector>
#include <initializer_list>
#include <stdint.h>
#include <cstdio>
#include <stdexcept>
#include <random>
#include <queue>
#include <iostream>
#include "NeuralNetwork.h"
#include "Data.h"
using namespace std;

/*MatDouble_t X {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0},
};*/

MatDouble_t X {{ 0.91319236,  0.43039519},
    {-0.48777018 , 0.92974249},
    {-0.12155395 ,-1.06319069},
    {-0.20430496 , 0.45936301},
    { 0.44224257 ,-0.06319802},
    { 0.43762974 ,-0.86416836},
    {-1.053644   , 0.0261938 },
    {-0.50678265 ,-0.87136511},
    {-0.31060856 , 0.41159227},
    {-1.00922551 ,-0.09350037},
    {-0.98703239 ,-0.05313556},
    {-0.53210781 ,-0.81144185},
    {-0.2207548  , 0.95688833},
    {-0.48536481 ,-0.05834583},
    {-0.17779105 , 0.43679391},
    {-0.59002223 ,-0.04163752},
    {-0.80169614 ,-0.67587024},
    { 0.66328897 , 0.01660735},
    { 0.09598078 ,-0.95810511},
    { 0.53019537 ,-0.74973896},
    {-0.37278314 ,-0.91107689},
    { 1.00577243 , 0.30599909},
    {-0.41126309 ,-0.36187441},
    { 0.11917118 ,-0.44812974},
    {-0.68257518 , 0.83419847},
    { 0.3144114  , 0.34691715},
    { 0.7405071  , 0.60538256},
    {-0.9853302  , 0.03033368},
    { 0.47530212 ,-0.7927544 },
    { 0.34515163 , 0.32401824},
    { 0.36039503 ,-0.98844067},
    {-0.04080144 , 0.50688876},
    { 0.94182186 ,-0.30354437},
    {-0.9848915  , 0.45072307},
    {-0.88583189 ,-0.64069083},
    { 0.60085081 , 0.8129415 },
    {-0.82933993 , 0.48595513},
    { 0.34078191 ,-0.34953378},
    { 0.47147638 , 0.0182687 },
    { 0.24833336 , 0.43300199},
    { 0.97242051 ,-0.28726736},
    { 0.57898871 ,-0.78639647},
    {-0.48905169 , 0.23104267},
    { 0.20158385 ,-0.4399962 },
    {-0.45301674 , 0.15357683},
    { 0.11173691 , 0.44413972},
    { 0.05385359 ,-0.4224208 },
    { 0.12185143 , 1.03896438},
    {-0.51732156 , 0.23507791},
    { 0.19169618 ,-0.45355898},
    {-0.01118127 , 0.52776507},
    {-0.84599207 , 0.35112472},
    { 0.42703108 , 0.20816469},
    { 0.29713146 ,-0.42432713},
    {-0.33476589 , 0.29878773},
    {-0.45863659 , 0.21896811},
    { 0.96803336 ,-0.42690518},
    { 0.37832252 ,-0.87864554},
    {-0.8866382  ,-0.35576817},
    {-0.83454788 ,-0.19831213},
    { 0.2964722  , 0.38842767},
    { 0.26373605 , 0.44992097},
    { 0.11143618 ,-0.94622419},
    {-0.12957454 , 0.48961953},
    { 0.50755585 ,-0.83692873},
    { 0.53261661 ,-0.01030856},
    {-0.30400023 ,-1.01207225},
    { 0.49236221 , 0.87179459},
    { 0.33511253 , 0.91666817},
    {-0.39580809 , 0.93261862},
    { 0.2455604  ,-0.43232924},
    { 0.47964399 , 0.82450916},
    { 0.98420017 ,-0.19592828},
    {-0.24026066 ,-0.38031732},
    {-0.88973185 , 0.49903945},
    {-0.46097904 ,-0.03499808},
    {-0.45122103 , 0.17709232},
    {-0.0399041  , 1.04724354},
    { 0.80213261 ,-0.4865392 },
    {-0.30563112 , 0.31882517},
    {-0.00730581 , 0.49079793},
    { 0.14726388 ,-0.54188548},
    { 0.23342641 ,-0.32690474},
    { 0.39595048 ,-0.3574525 },
    { 0.98436231 , 0.44022395},
    {-0.91798461 , 0.30248343},
    { 1.03125762 ,-0.04842096},
    {-0.94571679 ,-0.46430225},
    {-0.4772797  , 0.0902738 },
    {-0.10138172 , 1.00786585},
    {-0.87466923 , 0.55352778},
    {-0.07063177 ,-0.98709768},
    {-1.00229614 ,-0.25919273},
    {-0.01026059 ,-0.53308088},
    {-0.0712814  , 0.3455165 },
    {-0.40270372 ,-0.34771324},
    { 0.4883992  ,-0.248091  },
    { 0.14454157 , 0.47086933},
    {-0.98230636 , 0.27125857},
    { 0.75442834 ,-0.65259305},
    {-0.47446541 , 0.12166771},
    {-1.00164052 ,-0.23191232},
    {-0.02007823 , 0.53469604},
    {-0.31735499 ,-0.4005501 },
    {-0.44370884 , 0.10865834},
    { 0.2238624  , 0.47265658},
    {-0.2574889  , 0.34339272},
    {-0.95830272 , 0.32683155},
    {-0.48276726 ,-0.27238124},
    { 0.45287864 ,-0.06686971},
    {-0.60822753 ,-0.69703023},
    {-0.01878274 ,-0.51319298},
    {-0.12368116 ,-0.42664746},
    { 0.64218162 ,-0.71139221},
    {-0.06837793 , 0.44126602},
    {-0.5313345  , 0.88727822},
    { 0.89060639 ,-0.48278139},
    {-0.4362103  ,-0.03492163},
    {-0.25323432 , 0.42944633},
    {-0.00243763 , 0.98097861},
    { 0.37278767 , 0.37717028},
    {-0.36089474 , 0.28175742},
    {-1.04575781 ,-0.1835326 },
    {-0.30376676 ,-0.86603532},
    { 0.09781415 , 0.89290577},
    { 0.42343309 ,-0.925155  },
    {-0.89437937 , 0.4196145 },
    { 0.90749122 , 0.25022777},
    { 0.24403347 , 1.00835891},
    { 0.31657118 ,-0.38390342},
    { 0.10090084 ,-0.4261579 },
    { 0.17266088 , 1.04360378},
    {-0.48589392 , 0.06064096},
    {-0.59660326 ,-0.74995264},
    {-0.63929181 , 0.90336408},
    { 0.36827783 , 0.22464694},
    { 0.81954634 , 0.61132776},
    {-0.98971557 , 0.32375925},
    {-0.43114313 ,-0.39303938},
    { 0.73791416 , 0.61439272},
    {-0.28117095 ,-0.43836459},
    { 0.25324045 ,-0.33777326},
    {-0.5046603  , 0.20898495},
    { 0.2085888  , 0.43677872},
    { 0.09825694 ,-1.00432519},
    {-0.69333109 ,-0.73700202},
    { 0.36395122 , 0.19703846},
    { 0.238567   ,-0.42160386},
    { 0.29878976 ,-1.03671648},
    {-1.02285984 ,-0.1335967 },
    { 0.45243033 ,-0.18280247},
    {-0.96137397 , 0.10060424},
    {-0.37845851 , 0.41705295},
    { 0.39203351 , 0.40332493},
    { 0.47261321 , 0.11199979},
    {-0.76719558 , 0.63053711},
    { 0.36103896 , 0.26467914},
    {-0.50981815 ,-0.15134686},
    { 0.31828097 ,-0.24320583},
    { 0.47024348 ,-0.19740065},
    {-0.22392028 ,-0.42136183},
    {-0.91451162 ,-0.44924404},
    {-0.25139399 ,-0.94512059},
    {-0.4052328  , 0.27908713},
    { 0.0511741  , 0.50715776},
    { 0.99076162 ,-0.13360036},
    { 0.94533217 ,-0.03753303},
    { 1.00231991 ,-0.03606083},
    { 0.33855521 ,-0.47295387},
    {-0.80877553 , 0.45992191},
    {-0.38614813 ,-0.15767945},
    { 1.04153759 , 0.09493539},
    {-0.24804438 ,-0.32078433},
    { 0.10653763 ,-0.96829859},
    {-0.46809042 , 0.94074713},
    { 0.53504779 , 0.201198  },
    { 0.24141879 , 0.96089788},
    { 0.52972718 , 0.12158379},
    {-0.53691027 , 0.81056267},
    { 0.1283469  ,-0.5478404 },
    {-0.29682854 ,-1.01094395},
    {-0.94111055 ,-0.30942104},
    { 0.87934002 ,-0.47553195},
    {-0.10150091 ,-0.44587044},
    { 0.81869127 ,-0.55529158},
    {-0.28214657 , 0.83949818},
    { 0.42801261 ,-0.31127328},
    {-0.3367531  ,-0.38401363},
    {-0.36796906 ,-0.27278915},
    {-0.39174473 ,-0.16394888},
    { 0.8875125  , 0.48955683},
    { 0.54162458 ,-0.05141891},
    {-0.70612684 ,-0.69872329},
    { 0.98510419 ,-0.25416827},
    { 0.11510464 ,-1.03766587},
    { 0.42745156 , 0.46442439},
    { 0.93757513 , 0.4150902 },
    { 0.1704932  , 0.5516341 },
    { 0.35092246 , 0.95135446},
    { 0.53370478 , 0.82291583},
    { 0.15080773 ,-1.00959214},
    { 0.38856382 , 0.48896032},
    {-0.00639973 ,-1.00258643},
    { 0.47600723 , 0.04844896},
    { 0.87176407 , 0.05265138},
    {-0.31618967 , 0.92298714},
    {-0.91082631 ,-0.39530956},
    { 0.36855544 , 0.95504393},
    { 0.46763253 , 0.91432242},
    {-0.25173111 , 1.03505143},
    {-0.8795375  ,-0.18834028},
    {-0.46690527 , 0.89421971},
    { 0.76579615 , 0.6424153 },
    { 0.93291079 , 0.24614562},
    { 0.43143581 ,-0.23900438},
    { 0.46279938 ,-0.22143062},
    { 0.27800563 ,-0.37364768},
    { 0.44500906 , 0.24648673},
    {-1.03265677 , 0.25118281},
    { 0.74895806 , 0.62633325},
    {-0.48873947 , 0.01632451},
    { 0.19165123 ,-0.98163826},
    { 0.45405024 , 0.40329002},
    {-0.19759854 ,-0.50877812},
    {-0.05318967 , 1.04277   },
    { 0.39055267 , 0.13882636},
    { 0.40434597 , 0.88253671},
    {-0.08030855 ,-0.45141704},
    { 0.07527206 , 0.47572819},
    {-0.38788295 ,-0.90954317},
    {-0.31094662 ,-0.321337  },
    { 0.99632326 ,-0.17240152},
    { 0.4056775  ,-0.0638537 },
    { 0.59874713 ,-0.06313257},
    { 0.37822953 , 0.1323923 },
    {-0.16536337 , 0.96693352},
    { 0.87744647 , 0.50712007},
    {-0.34244461 , 0.89856124},
    {-0.13940518 , 0.48199592},
    { 0.16710835 ,-0.49023054},
    { 0.97885551 ,-0.3699266 },
    {-0.33874105 , 0.37031478},
    { 0.26903974 , 0.32084977},
    { 0.16334274 , 0.93679182},
    {-0.96116778 ,-0.04232387},
    { 0.85921634 ,-0.452965  },
    {-0.51731316 , 0.06231936},
    { 0.57403656 , 0.89342453},
    { 1.02316296 , 0.04341579},
    {-0.9826561  , 0.12199446},
    {-0.51434161 ,-0.91599813},
    {-0.94120967 ,-0.32666627},
    {-0.2856589  ,-0.44299315},
    {-0.10696273 , 0.99446093},
    {-0.24843224 , 0.47774022},
    {-0.89790656 , 0.38064007},
    { 0.59306397 , 0.05494722},
    { 0.32989178 , 0.33254552},
    { 0.44124027 ,-0.89871198},
    {-0.8748531  ,-0.54172369},
    { 1.01169243 , 0.07556963},
    {-0.49174269 , 0.15541096},
    {-0.17339192 ,-0.50791675},
    { 0.42467605 , 0.0556776 },
    {-0.90110042 , 0.11581248},
    {-0.39822095 , 0.9780242 },
    {-0.38937559 ,-0.29349796},
    {-0.88436445 ,-0.68872961},
    { 0.36727771 ,-0.33373551},
    {-0.66933536 ,-0.86162255},
    {-0.34661502 , 0.35358005},
    { 0.45379205 , 0.15592654},
    { 0.0323568  ,-0.55550496},
    {-0.20537141 , 0.54161861},
    {-1.00653094 ,-0.44915004},
    { 0.18018638 , 0.50314678},
    { 0.15660432 , 0.40207527},
    { 0.30985045 ,-0.96574631},
    { 0.55778569 , 0.86937711},
    { 0.2519146  ,-0.39769022},
    {-0.2879437  ,-0.46019217},
    {-0.29678873 , 0.42533536},
    { 0.95568605 ,-0.52191441},
    { 1.00556992 ,-0.12155364},
    {-0.46756142 ,-0.1293478 },
    {-0.23860464 , 0.45780712},
    {-0.20567392 , 0.42231879},
    { 0.52558261 , 0.12127381},
    { 0.70144769 , 0.64993528},
    { 0.93375018 ,-0.31464419},
    { 0.48552675 , 0.02480013},
    {-0.5633929  ,-0.86138866},
    {-0.77627698 ,-0.59825441},
    {-0.44047916 ,-0.05181599},
    {-0.6076779  ,-0.10954906},
    {-0.62325072 , 0.6774765 },
    { 0.54787103 , 0.12628517},
    { 0.66607994 ,-0.64181813},
    {-0.52652134 , 0.76793609},
    { 0.03223253 ,-1.04413715},
    {-0.5794151  , 0.83320246},
    {-0.13693345 , 0.48071321},
    { 0.56953083 , 0.77526785},
    { 0.65768548 ,-0.74908509},
    {-0.16907943 , 1.00156731},
    { 0.67970231 ,-0.66976941},
    { 0.54239275 ,-0.05572674},
    { 0.3168127  ,-0.52160502},
    { 0.43402233 ,-0.29915884},
    { 0.97922935 , 0.355506  },
    { 0.2973519  ,-0.9764203 },
    { 0.62964987 ,-0.79217227},
    { 0.26258276 ,-0.48303031},
    { 0.89669096 ,-0.38387435},
    {-0.04586621 ,-0.44189936},
    {-0.43059285 , 0.32273652},
    {-0.3421729  ,-0.31782324},
    {-0.72512957 , 0.73100991},
    {-0.08816481 , 0.50509703},
    { 0.24214515 , 0.44098992},
    { 0.03046242 , 0.99756307},
    { 0.80174043 ,-0.57257283},
    {-0.7285026  , 0.70486351},
    { 0.95429986 , 0.24667082},
    { 0.92637311 , 0.41000044},
    {-0.47428292 , 0.00566382},
    { 0.94348095 ,-0.09223183},
    {-0.39066854 ,-0.17678551},
    { 0.50020502 ,-0.8865327 },
    {-0.62348789 , 0.72886792},
    { 0.06757075 , 0.40169206},
    { 0.02826965 ,-0.54929462},
    {-0.35179285 ,-0.24867898},
    { 0.50833084 , 0.76806125},
    {-0.47347943 , 0.86666816},
    {-0.18834955 ,-0.97703455},
    {-0.38635958 , 0.28966368},
    {-0.88738599 ,-0.50804884},
    {-0.6609168  ,-0.66711726},
    {-0.12681828 ,-0.46384588},
    {-0.43049839 , 0.06486105},
    { 0.06568822 ,-0.5589225 },
    {-0.51208516 , 0.30140261},
    { 0.24459162 ,-0.98092961},
    { 0.24428407 , 0.33811696},
    {-0.21604684 , 0.90637544},
    {-0.47524596 ,-0.14526132},
    {-0.17391602 ,-0.42119386},
    {-0.511159   , 0.0110003 },
    {-0.4047653  ,-0.23049801},
    { 0.50192356 ,-0.20830916},
    { 0.33793608 ,-0.31017076},
    {-0.48648408 , 0.24985112},
    {-0.49077175 ,-0.24537016},
    {-1.02596447 ,-0.15741285},
    { 0.04623535 , 0.97249773},
    {-0.50754012 ,-0.14829087},
    { 0.36342058 ,-0.06263939},
    {-0.77843249 , 0.60828706},
    { 0.76266439 , 0.59867179},
    {-0.51998117 ,-0.31063687},
    { 0.4316115  ,-0.14445486},
    { 0.47111468 , 0.20959695},
    { 0.37412453 , 0.39849761},
    { 0.94560011 , 0.23442212},
    {-0.48640403 , 0.09559597},
    { 0.01836449 , 0.49932781},
    { 0.38599188 , 0.32456217},
    { 0.47078822 ,-0.22512016},
    {-0.65742445 , 0.64320652},
    { 0.51807285 ,-0.10745695},
    { 0.74257544 , 0.76761185},
    {-0.9476636  ,-0.19622544},
    { 0.90025991 , 0.53943886},
    {-0.02865052 , 0.9615396 },
    { 0.54511423 , 0.05341627},
    { 0.69084387 ,-0.76385613},
    {-0.21248112 ,-0.51493346},
    {-0.01227419 ,-0.55867338},
    { 0.47303247 , 0.08422244},
    { 0.26164092 , 1.03913686},
    {-0.28883773 ,-0.48964732},
    {-0.34021745 ,-0.88689319},
    { 0.62658967 , 0.72848558},
    {-0.1913369  ,-0.97525851},
    {-0.41621872 ,-0.91840288},
    { 0.86800358 ,-0.63088801},
    {-0.99731737 , 0.03268492},
    { 0.51792954 , 0.25349493},
    { 0.39774897 , 0.86578516},
    { 0.2290579  , 0.49316736},
    {-0.12011199 , 0.47359087},
    {-0.43799397 , 0.88982967},
    { 0.51458189 ,-0.87545232},
    {-0.44834466 , 0.0134259 },
    {-0.42560064 ,-0.22866137},
    { 0.63349459 ,-0.84378754},
    { 0.48742446 , 0.16190697},
    {-0.03307823 , 0.47155017},
    { 0.63581124 ,-0.77479178},
    { 0.08917256 ,-0.44763131},
    { 1.02312108 , 0.19557488},
    { 0.51175498 ,-0.262322  },
    {-0.0999899  ,-0.52783204},
    { 0.29846343 , 0.34026416},
    { 0.30069104 , 0.39598019},
    {-0.94081344 , 0.4302476 },
    {-0.01586115 , 0.43673176},
    { 0.02623352 ,-0.485478  },
    { 0.40214559 , 0.32178442},
    {-0.74591185 , 0.78653673},
    {-0.10848767 ,-1.02413765},
    {-0.05538753 , 0.55775341},
    { 0.35041537 ,-0.40599312},
    {-0.33320213 ,-0.9457648 },
    { 0.90234543 ,-0.47371023},
    { 0.44760753 ,-0.21035889},
    { 0.65342815 , 0.80440917},
    {-0.65700078 ,-0.73400968},
    { 0.97143788 ,-0.03152354},
    { 0.97889152 ,-0.18153494},
    {-0.01580031 , 0.51659476},
    { 0.2537241  , 0.43339366},
    { 0.39524775 , 0.91680918},
    {-0.29795172 ,-0.44367815},
    {-0.19780036 ,-0.405618  },
    {-1.00354808 ,-0.42827844},
    { 0.74901438 ,-0.65349248},
    {-0.41800279 , 0.12055804},
    {-0.10192536 ,-1.01295783},
    {-0.23783681 , 0.48944913},
    { 0.18722158 ,-0.50658635},
    { 0.43305712 ,-0.13674839},
    {-0.61585217 ,-0.78584619},
    {-0.30964483 , 0.4167579 },
    {-0.33099784 ,-0.36996461},
    { 0.62540956 ,-0.73466158},
    {-0.85466937 , 0.55505005},
    {-0.48981312 ,-0.06297219},
    { 0.93359354 ,-0.32718827},
    { 0.30369275 ,-0.41139748},
    {-0.95912636 , 0.11781168},
    {-0.37727721 , 0.37937175},
    { 0.99588825 , 0.03892692},
    {-0.97399957 , 0.29976391},
    { 0.22709717 ,-0.294248  },
    {-0.34143394 ,-0.93227442},
    { 0.52168464 , 0.06768592},
    { 0.08369806 ,-0.41129135},
    { 0.12998519 ,-0.5253064 },
    {-0.11824717 ,-1.00476736},
    {-0.48381575 ,-0.79344947},
    { 0.41854902 ,-0.21920977},
    {-0.51262632 , 0.07505708},
    { 0.38345942 , 0.21402159},
    { 0.05754144 ,-0.86518373},
    { 0.96500782 ,-0.03742568},
    { 0.15741126 , 0.97792492},
    {-0.51365762 , 0.02305103},
    { 0.74666554 , 0.62958548},
    { 0.52532769 ,-0.9148809 },
    {-0.43209065 ,-0.07869874},
    {-0.29320353 ,-0.19964509},
    { 0.32863901 , 0.95023955},
    {-0.44363876 , 0.29947405},
    { 0.81164234 , 0.44589667},
    { 0.26309081 , 0.52146098},
    {-0.34050001 ,-0.8330497 },
    {-0.25337104 , 0.41725638},
    {-0.42783859 ,-0.00237163},
    { 0.74636186 , 0.70237088},
    {-0.43660629 ,-0.9602895 },
    { 0.9204552  , 0.18224341},
    {-0.08569017 , 0.52055846},
    {-0.74076282 ,-0.62903716},
    {-0.83464824 , 0.36459234},
    {-0.32031064 ,-0.31721647},
    {-0.75490463 , 0.56250211},
    {-0.94940993 , 0.43379293},
    {-0.51979216 , 0.22580665},
    {-0.13293437 , 0.97928089},
    {-0.29893827 , 0.28783359},
    {-0.06343006 ,-0.52572077},
    {-1.01663091 ,-0.10633483},
    {-0.01132126 , 0.47915776},
    { 0.08393711 ,-0.40094832},
    { 0.48888701 , 0.10833963},
    {-0.04229791 , 0.51544708},
    {-0.73869985 ,-0.55624865},
    { 0.3828924  ,-0.15963167},
    {-0.23953458 , 0.40225945},
    { 0.7928984  , 0.61242359},
    {-0.36836551 ,-0.32221697},
    {-0.15238426 ,-0.53863059},
    {-0.70493952 , 0.7157937 },
    {-0.23765568 ,-0.55917837},
    {-0.82979143 ,-0.69226335},
    { 0.4561244  ,-0.28495257},
    {-0.2063582  ,-0.38724486},
    {-1.04152175 , 0.14525613}};

/*MatDouble_t Y {
    {0.0,0.0},
    {1.0,1.1},
    {1.0,1.1},
    {0.0,0.0}
};*/

MatDouble_t Y {
        {0},{0},{0},{1},{1},{0},{0},{0},{1},{0},{0},{0},{0},{1},{1},{1},{0},{1},{0},{0},{0},{0},{1},{1},{0},{1},{0},{0},{0},{1},{0},{1},{0},{0},{0},{0},{0}
        ,{1},{1},{1},{0},{0},{1},{1},{1},{1},{1},{0},{1},{1},{1},{0},{1},{1},{1},{1},{0},{0},{0},{0},{1},{1},{0},{1},{0},{1},{0},{0},{0},{0},{1},{0},{0},{1}
        ,{0},{1},{1},{0},{0},{1},{1},{1},{1},{1},{0},{0},{0},{0},{1},{0},{0},{0},{0},{1},{1},{1},{1},{1},{0},{0},{1},{0},{1},{1},{1},{1},{1},{0},{1},{1},{0}
        ,{1},{1},{0},{1},{0},{0},{1},{1},{0},{1},{1},{0},{0},{0},{0},{0},{0},{0},{1},{1},{0},{1},{0},{0},{1},{0},{0},{1},{0},{1},{1},{1},{1},{0},{0},{1},{1}
        ,{0},{0},{1},{0},{1},{1},{1},{0},{1},{1},{1},{1},{1},{0},{0},{1},{1},{0},{0},{0},{1},{0},{1},{0},{1},{0},{0},{1},{0},{1},{0},{1},{0},{0},{0},{1},{0}
        ,{0},{1},{1},{1},{1},{0},{1},{0},{0},{0},{1},{0},{1},{0},{0},{0},{1},{0},{1},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{1},{1},{1},{1},{0},{0},{1},{0}
        ,{1},{1},{0},{1},{0},{1},{1},{0},{1},{0},{1},{1},{1},{0},{0},{0},{1},{1},{0},{1},{1},{0},{0},{0},{1},{0},{0},{0},{0},{0},{1},{0},{1},{0},{1},{1},{0}
        ,{0},{0},{1},{1},{1},{0},{0},{1},{0},{1},{0},{1},{1},{1},{1},{0},{1},{1},{0},{0},{1},{1},{1},{0},{0},{1},{1},{1},{1},{0},{0},{1},{0},{0},{1},{1},{0}
        ,{1},{0},{0},{0},{0},{1},{0},{0},{0},{0},{1},{1},{1},{0},{0},{0},{1},{0},{1},{1},{1},{0},{1},{1},{0},{0},{0},{0},{0},{1},{0},{1},{0},{0},{1},{1},{1}
        ,{0},{0},{0},{1},{0},{0},{1},{1},{1},{1},{0},{1},{0},{1},{1},{1},{1},{1},{1},{1},{1},{0},{0},{1},{1},{0},{0},{1},{1},{1},{1},{0},{1},{1},{1},{1},{0}
        ,{1},{0},{0},{0},{0},{1},{0},{1},{1},{1},{0},{1},{0},{0},{0},{0},{0},{0},{1},{0},{1},{1},{0},{0},{1},{1},{0},{1},{1},{0},{1},{0},{1},{1},{1},{1},{0}
        ,{1},{1},{1},{0},{0},{1},{1},{0},{0},{1},{0},{0},{0},{0},{1},{1},{0},{1},{1},{0},{0},{1},{0},{1},{1},{1},{0},{1},{1},{0},{0},{1},{0},{1},{0},{1},{0}
        ,{0},{1},{0},{1},{1},{1},{0},{0},{1},{1},{1},{0},{0},{0},{1},{0},{0},{1},{1},{0},{1},{0},{1},{0},{1},{1},{0},{0},{0},{1},{0},{0},{1},{0},{0},{1},{0}
        ,{1},{1},{0},{1},{1},{1},{1},{0},{1},{1},{0},{1},{1},{0},{1},{0},{1},{1},{0}
    };

/*void randomTrain(initializer_list<uint16_t> layerStruct, NeuralNetwork_t& bestNet,MatDouble_t const& X, VecDouble_t const& Y,float learningR){
    double lessFails=pow(X.size(),2);
    double fails=0;


    for(size_t i=0;i<5000;i++){
        //Iteracion i
        NeuralNetwork_t net(layerStruct,learningR);
        
        fails=net.evaluateNet(X,Y);
        //Si es la mejor la guardamos
        if(fails<lessFails){
            lessFails=fails;
            bestNet=net;
        }

    }
}*/

MatDouble_t vectorOfVectorsToMatDouble(const auto &vv){
    MatDouble_t m;

    for(size_t i=0;i<vv.size();i++){
        vector<double> vd;
        for(size_t j=0;j<vv[i].size();j++){
            vd.push_back(vv[i][j]);
        }
        m.push_back(vd);
    }

    return m;
}

void splitDataTrainTest(double percent,MatDouble_t const& X,MatDouble_t const& Y,
                        MatDouble_t& Xtrain,MatDouble_t& Ytrain,
                        MatDouble_t& Xval,MatDouble_t& Yval){
    size_t i=0;

    Xtrain.resize(0);
    Ytrain.resize(0);
    Xval.resize(0);
    Yval.resize(0);
    cout << "Separando train data..." << endl;
    while(i<(percent*X.size())){
        Xtrain.push_back(X[i]);
        Ytrain.push_back(Y[i]);
        i++;
    }
    cout << "Separando validation data..." << endl;
    while(i<X.size()){
        Xval.push_back(X[i]);
        Yval.push_back(Y[i]);
        i++;
    }
    cout << "Datos separados" << endl;
}

void run(){
    //Leemos los datos
    Data dataTrain;
    dataTrain.init("data/dataTrainScaled255.csv",59,5);
    Data dataVal;
    dataVal.init("data/dataValScaled255.csv",59,5);

    cout << "Numero de datos:" << endl;
    for(size_t i=0;i<dataTrain.Yneg.size();i++){
        cout << "Neurona de salida " << i << endl;
        cout << "Negativos: " << dataTrain.Yneg[i] << endl;
        cout << "Positivos: " << dataTrain.Ypos[i] << endl;
        cout << "Totales: " << dataTrain.Y.size() << endl << endl;
    }


    //Generamos la red neuronal
    //Antes: initializer_list<uint16_t> layerStruct={dataTrain.tamXi,64,32,16,dataTrain.tamYi}; //Mejor resultado
    initializer_list<uint16_t> layerStruct={dataTrain.tamXi,64,32,16,dataTrain.tamYi};  
    //initializer_list<uint16_t> layerStruct={X[0].size(),4,8,Y[0].size()};
    float learningRate=0.06;
    uint16_t epochs=500;
    uint16_t patience=5;

    NeuralNetwork_t net(layerStruct,learningRate);

    //net.setActiveFunctions({ActF::RELU,ActF::RELU,ActF::RELU,ActF::SIGMOID});

    cout << "Datos mínimos necesarios: ";
    uint16_t minData=0;
    for(size_t i=0;i<net.getm_layers().size();i++){
        minData+=net.getm_layers()[i].size()*net.getm_layers()[i][0].size();
    }
    minData=minData*10;
    cout << minData << endl << endl;

    //MatDouble_t Xtrain,Ytrain,Xval,Yval;
    //splitDataTrainTest(0.9,vectorOfVectorsToMatDouble(data.X),vectorOfVectorsToMatDouble(data.Y),
    //                    Xtrain,Ytrain,Xval,Yval);

    //Entrenamos la red neuronal
    //net.train(Xtrain,Ytrain,Xval,Yval,10);
    net.train(vectorOfVectorsToMatDouble(dataTrain.X),
              vectorOfVectorsToMatDouble(dataTrain.Y),
              vectorOfVectorsToMatDouble(dataVal.X),
              vectorOfVectorsToMatDouble(dataVal.Y),
              epochs,
              patience);
    //net.train(X,Y,250);
    cout << "Fallos:" << endl;
    MatDouble_t fallos=net.test(vectorOfVectorsToMatDouble(dataVal.X),vectorOfVectorsToMatDouble(dataVal.Y));
    double fallosSesgados=0;
    double fallosNoSesgados=0;

    for(size_t i=0;i<fallos.size();i++){
        /*cout << "Por moverse cuando no debia: " << endl;
        cout << fallos[i][0] << "/" << dataVal.Yneg[i] << "->";
        cout << (double)fallos[i][0]/dataVal.Yneg[i]*100 << "%" << " -> ";
        cout << "Sesgado: " << (double)fallos[i][0]/dataVal.Yneg[i]*100 * dataVal.Ypos[i] / dataVal.Y.size() << endl;

        cout << "Por no moverse cuando debia: "<< endl;
        cout << fallos[i][1] << "/" << dataVal.Ypos[i] << "->";
        cout << (double)fallos[i][1]/dataVal.Ypos[i]*100 << "%" << " -> ";
        cout << "Sesgado: " << (double)fallos[i][1]/dataVal.Ypos[i]*100 * dataVal.Yneg[i] / dataVal.Y.size() << endl;

        cout << "Totales: "<< endl;
        cout << (fallos[i][1]+fallos[i][0]) << "/" << dataVal.Y.size() << "->";
        cout << (double)(fallos[i][1]+fallos[i][0])/dataVal.Y.size()*100 << "%" << " -> ";
        cout << "Sesgado: " << (double)fallos[i][0]/dataVal.Yneg[i]*100 * dataVal.Ypos[i] / dataVal.Y.size()+(double)fallos[i][1]/dataVal.Ypos[i]*100 * dataVal.Yneg[i] / dataVal.Y.size()/2 << endl;
        cout << endl;*/
        double fallosSesgadosAux=0;
        double porcentajeFallosAux =(double)fallos[i][0]/dataVal.Yneg[i]*100; //Porcentaje de fallos negativos
        fallosSesgadosAux += porcentajeFallosAux * dataVal.Ypos[i] / dataVal.Y.size();   //Sesgo

        porcentajeFallosAux = (double)fallos[i][1]/dataVal.Ypos[i]*100; //Porcentaje de fallos positivos
        fallosSesgadosAux += porcentajeFallosAux * dataVal.Yneg[i] / dataVal.Y.size();   //Sesgo

        fallosSesgadosAux/=2;   //Media

        fallosSesgados+=fallosSesgadosAux;
        fallosNoSesgados+=(double)(fallos[i][1]+fallos[i][0])/dataVal.Y.size()*100;
    }

    fallosSesgados/=fallos.size();
    fallosNoSesgados/=fallos.size();

    cout << "Porcentaje total no sesgado de fallos: " << fallosNoSesgados << endl;
    cout << "Porcentaje total sesgado de fallos: " << fallosSesgados<< endl;

    net.save("models/NeuralNetwork.txt");
    /*net.load("NeuralNetwork2.txt");

    net.feedforward(vectorOfVectorsToMatDouble(data.X)[0]);*/


    //Predecimos los valores
    /*auto res = net.feedforward({0.0 , 0.0});
    printf("0.0 xor 0.0 = %f -> %i\n",res[0],(int)(res[0]+0.5));
    res = net.feedforward({0.0 , 1.0});
    printf("0.0 xor 1.0 = %f -> %i\n",res[0],(int)(res[0]+0.5));
    res = net.feedforward({1.0 , 0.0});
    printf("1.0 xor 0.0 = %f -> %i\n",res[0],(int)(res[0]+0.5));
    res = net.feedforward({1.0 , 1.0});
    printf("1.0 xor 1.0 = %f -> %i\n",res[0],(int)(res[0]+0.5));*/
}

int main(){

    try{
        run();
    }catch(exception &e){
        printf("[EXCEPTION]: %s\n",e.what());
        return -1;
    }

    return 0;
}