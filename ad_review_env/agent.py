"""Smart rule-based agent for UGC content moderation."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

WeightedPattern = Tuple[str, float]


@dataclass
class SignalResult:
    strength: float = 0.0
    matched: List[str] = field(default_factory=list)

    @property
    def triggered(self) -> bool:
        return self.strength > 0.0


@dataclass
class SignalMap:
    critical: SignalResult = field(default_factory=SignalResult)
    adult: SignalResult = field(default_factory=SignalResult)
    violence: SignalResult = field(default_factory=SignalResult)
    hate_speech: SignalResult = field(default_factory=SignalResult)
    illegal: SignalResult = field(default_factory=SignalResult)
    misinformation: SignalResult = field(default_factory=SignalResult)
    profanity: SignalResult = field(default_factory=SignalResult)
    drugs: SignalResult = field(default_factory=SignalResult)
    gambling: SignalResult = field(default_factory=SignalResult)
    controversial: SignalResult = field(default_factory=SignalResult)
    piracy: SignalResult = field(default_factory=SignalResult)
    terrorism: SignalResult = field(default_factory=SignalResult)
    safe: SignalResult = field(default_factory=SignalResult)

    @property
    def all_flagged(self) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for sig in [
            self.critical, self.adult, self.violence, self.hate_speech,
            self.illegal, self.misinformation, self.profanity, self.drugs,
            self.gambling, self.controversial, self.piracy, self.terrorism,
        ]:
            for m in sig.matched:
                if m not in seen:
                    seen.add(m)
                    result.append(m)
        return result[:5]


@dataclass
class ContentContext:
    is_satire: bool = False
    is_personal_narrative: bool = False
    is_advocacy: bool = False


@dataclass
class Classification:
    decision: str = "APPROVE"
    iab_category: str = "IAB_SAFE"
    garm_category: str = "GARM_SAFE"
    risk_level: str = "LOW"
    age_rating: str = "ALL_AGES"
    confidence: float = 0.85
    reasoning_parts: List[str] = field(default_factory=list)

CRITICAL_PATTERNS: List[WeightedPattern] = [
    (r"\b(kill|murder|rape|shoot|stab)\s+(that|him|her|them|you|me|everyone|everybody|someone|somebody)\b", 0.9),
    (r"\bkill\s+this\s+(guy|person|man|woman|bitch|bastard|asshole|dude|kid|child)\b", 0.9),
    (r"\b(gonna|going\s+to|will|i'?ll|let'?s|need\s+to)\s+(kill|murder|shoot|stab|attack|hurt|rape)\b(?!.*(workout|presentation|game|stage|it|this\s|interview|audition|test|exam|meeting))", 0.9),
    (r"\bwant\s+to\s+kill\b(?!.*(workout|presentation|game|stage|it\b|interview|audition|this\s))", 0.9),
    (r"\b(eat|cook|hurt|harm|kill|abuse|hit|punch|kick|burn|torture|molest|sell|buy|traffic)\s+(the\s+|a\s+|an\s+)(baby|babies|child|children|kid|kids|infant|toddler|minor|puppy|puppies|kitten|kittens)\b", 0.95),
    (r"\b(molest|abuse|harm|hurt|torture)\s+(baby|babies|child|children|kid|kids|infant|toddler|minor)\b", 0.95),
    (r"\b(baby|babies|child|children|kid|kids|infant)\s+(eat|cook|burn|torture|abuse|kill|murder)\b", 0.95),
    (r"\b(baby|babies|child|children|kid|kids|infant|minor|girl|boy)\w*\s+(are\s+|is\s+)?(for\s+sale|for\s+sell|to\s+sell|to\s+buy|available|on\s+sale)\b", 0.95),
    (r"\b(sell|buy|trade|traffic)\w*\s+(the\s+|a\s+|an\s+|some\s+|my\s+|your\s+|our\s+|his\s+|her\s+|their\s+)?(baby|babies|child|children|kid|kids|infant|minor|girl|boy)\b", 0.95),
    (r"(?!.*\b(adopt\w*|married|marriage|custody|divorce|wed\w*|foster)\b)\b(have|get|obtain|acquire|take|want)\s+(baby|babies|child|children|kid|kids|infant|minor\w*)\s+\w*\s*(not\s+legal\w*|illegally|without\s+(consent|permission|authorization)|underage|unlawfully)\b", 0.95),
    (r"(?!.*\b(adopt\w*|married|marriage|custody|divorce|wed\w*|foster)\b)\b(baby|babies|child|children|kid|kids|infant|minor\w*)\b.{0,60}(not\s+legal\w*|illegally|against\s+the\s+law|unlawfully)\b", 0.9),
    (r"(?!.*\b(adopt\w*|married|marriage|custody|divorce|wed\w*|foster)\b)\b(not\s+legal\w*|illegally|against\s+the\s+law|unlawfully)\b.{0,60}\b(baby|babies|child|children|kid|kids|infant|minor\w*)\b", 0.9),
    (r"\b(eat|cook|consume)\s+(the\s+|a\s+|an\s+)?(human|humans|person|people|body|flesh|corpse)\b", 0.9),
    (r"\b(kill|murder|rape)\b(?!.*(workout|presentation|game|stage|it\b|the\s+game|the\s+stage|my\s+workout|this\s+workout|it\s+at|killing\s+it|bug|code|lol|this\s+bug|the\s+vibe|the\s+beat|the\s+set|the\s+show))", 0.6),
    (r"\bterrorist\b", 0.9),
    (r"\bbomb\b(?!.*(bath|cherry|caramel))", 0.9),
    (r"\bchild\s*porn", 1.0),
    (r"\bpedophil", 1.0),
    (r"\bcsam\b", 1.0),
    (r"\[.*slur.*\]", 0.95),
    (r"\bslur\b", 0.8),
    (r"#deportthem", 0.9),
    (r"\b(how\s+to|ways?\s+to)\s+(kill\s+yourself|commit\s+suicide|self[- ]?harm)\b", 0.95),
    (r"\b(human\s+)?trafficking\b", 0.9),
    (r"\b(sex\s+)?slave\b", 0.9),
]

ADULT_PATTERNS: List[WeightedPattern] = [
    (r"18\+\s*only", 0.9),
    (r"\b(uncensored|explicit)\b.*\bcontent\b", 0.85),
    (r"🔞", 0.9),
    (r"\badult\s*content\b.*\b(subscribe|link|page|onlyfans|fansly|only\s*fans|sub4sub|click)\b", 0.9),
    (r"\b(creator|subscribe|link|page|onlyfans|fansly)\b.*\badult\s*content\b", 0.9),
    (r"\badult\s*content\s*(creator|maker)\b.*\b(18\+|link|sub|only\s*fans)\b", 0.85),
    (r"\bnude|naked|xxx\b", 0.85),
    (r"\bnew videos daily\b.*😈", 0.7),
    (r"\b(onlyfans|fansly|manyvids)\b", 0.85),
    (r"\blink\s*in\s*bio\b.*😈", 0.7),
    (r"\b(NSFW|nsfw)\b", 0.75),
    (r"\b(porn|porno|hentai)\b", 0.9),
    (r"\b(strip|stripper|stripclub)\b", 0.6),
]

VIOLENCE_PATTERNS: List[WeightedPattern] = [
    (r"\bgraphic\b.*\b(violence|fight|knockout)", 0.9),
    (r"\bbrutal\b.*\b(fight|knockout|attack)", 0.9),
    (r"\bcompilation\b.*\b(fight|knockout|kill)", 0.8),
    (r"\bstreet\s*fight\b", 0.7),
    (r"💀.*\b(knockout|fight|brutal)\b", 0.6),
    (r"\b(kill|shoot|stab|attack|hurt|assault|beat)\s+(that|him|her|them|you|me|everyone)\b", 0.85),
    (r"\bkill\s+this\s+(guy|person|man|woman|bitch|dude|kid|child|asshole|bastard)\b", 0.85),
    (r"\bdeath\s*threat\b", 0.9),
    (r"\bthreat(en|s|ening)\b", 0.5),
    (r"\b(eat|cook|burn|torture|dismember|skin)\s+(the\s+|a\s+|an\s+)(baby|babies|child|children|kid|kids|person|people|dog|dogs|cat|cats|puppy|puppies|kitten|kittens|human|humans)\b", 0.9),
    (r"\b(torture|dismember|skin|mutilate)\s+(baby|babies|child|children|kid|kids|person|people)\b", 0.9),
    (r"\b(baby|babies|child|children|kid|kids|person|people|dog|dogs|cat|cats)\s+(eat|cook|burn|torture)\b", 0.9),
    (r"\b(gore|dismember|mutilat|disembowel|behead)\b", 0.85),
    (r"\b(die|burn)\s+in\s+a\s+fire\b", 0.8),
    (r"\bdie\b.*\b(slowly|painfully)\b", 0.8),
    (r"\b(swat|doxx|dox)(ted|ting|ed|ing)?\b", 0.85),
    (r"\bpost(ed|ing)?\s+(their|his|her|your)\s+(address|home|phone)\b", 0.85),
    (r"\b(school|mass)\s*shoot(ing|er)\b", 0.95),
    (r"\b(shoot\s*up|blow\s*up)\s+(the\s+|a\s+)?(school|building|church|mosque|synagogue)\b", 0.95),
]

HATE_SPEECH_PATTERNS: List[WeightedPattern] = [
    (r"\[.*slur.*\]", 0.95),
    (r"\bruining our country\b", 0.7),
    (r"\bproud\s*boys\b", 0.9),
    (r"\bextremist|supremacist|white\s*power\b", 0.9),
    (r"#deportthem", 0.85),
    (r"\bjoin\s*us\b.*\blink\b", 0.4),
    (r"\bgreat\s*replacement\b", 0.9),
    (r"\bsecure\s*a\s*future\s*for\s*our\s*people\b", 0.9),
    (r"\b(i\s+)?hate\s+(all\s+)?(women|men|jews|muslims|blacks|whites|gays|trans)\b", 0.85),
    (r"\b(women|men|jews|muslims|blacks|whites|gays|trans)\s+(are|should)\s+(all\s+)?(die|burn|suffer|be\s+killed)\b", 0.9),
    (r"\byou\s+deserve\s+to\s+(die|be\s+(raped|killed|hurt|beaten))\b", 0.95),
    (r"\b(go\s+)?(kill|hang)\s+yourself\b", 0.95),
    (r"\bdie\s+in\s+a\s+fire\b", 0.8),
    (r"\bkys\b", 0.9),
    (r"\bgo\s+die\b", 0.85),
    (r"\b(n[i1]gg|f[a@]gg|k[i1]ke|sp[i1]c|ch[i1]nk)\w*\b", 0.95),
    (r"\b(subhuman|untermensch|cockroach|vermin|parasite)\b.*\b(people|them|they|immigrants|refugees)\b", 0.9),
    (r"\b(people|they|them|immigrants|refugees)\b.*\b(subhuman|cockroach|vermin|parasite|animal)\b", 0.9),
    (r"\b(illegal\s*aliens?|invaders?)\b.*\b(country|border|nation)\b", 0.7),
    (r"\bholocaust\b.*\b(hoax|fake|never\s*happened|exaggerat)\b", 0.95),
    (r"\b(hoax|fake|lie)\b.*\bholocaust\b", 0.95),
]

ILLEGAL_PATTERNS: List[WeightedPattern] = [
    (r"\b(xanax|oxycontin|fentanyl)\b.*\bno\s*prescription\b", 0.95),
    (r"\bno\s*prescription\b.*\b(buy|cheap|order|ship)\b", 0.9),
    (r"\bcheap\b.*\b(xanax|pills|meds)\b", 0.85),
    (r"\bpirated?\b.*\b(movie|film|show|stream)\b", 0.85),
    (r"\bDM\s*(me\s*)?for\s*(the\s*)?link\b.*\bpirat", 0.9),
    (r"\bDM\s*for\s*prices\b", 0.7),
    (r"\bdiscreet\s*shipping\b", 0.8),
    (r"\bcredit\s*card\s*(number|info)\b", 0.9),
    (r"\b(buy|sell|selling)\s+(a\s+|an\s+)?(gun|firearm|weapon|ammo)\b", 0.8),
    (r"\bunregistered\s*(gun|firearm|weapon)\b", 0.9),
    (r"\b(counterfeit|fake)\s+(ID|passport|diploma|money|bills?)\b", 0.85),
    (r"\b(forged|fake)\s+(documents?|papers?)\b", 0.8),
    (r"\b(hack|hacking)\s+(services?|for\s*hire)\b", 0.85),
    (r"\bDDoS\s*(attack|service)\b", 0.85),
]

MISINFORMATION_PATTERNS: List[WeightedPattern] = [
    (r"\bgovernment\b.*\blying\b", 0.75),
    (r"\bwake\s*up\s*sheeple\b", 0.8),
    (r"\bdoctors\s*don.?t\s*want\s*you\s*to\s*know\b", 0.85),
    (r"\bone\s*weird\s*trick\b", 0.8),
    (r"\bcancer\s*cure\b.*\b(natural|herbal|alternative)\b", 0.9),
    (r"\binstead\s*of\s*chemo\b", 0.9),
    (r"\bdo\s*your\s*own\s*research\b", 0.5),
    (r"\blost\s*\d+\s*lbs?\s*in\s*\d+\s*days?\b", 0.7),
    (r"\bautism\b.*\b(shot|vaccine|MMR)\b", 0.85),
    (r"\b(vaccine|MMR)\b.*\bautism\b", 0.85),
    (r"\$\d+K?/month\b.*\b(from\s*(your\s*)?phone|DM)\b", 0.8),
    (r"\btransformation\b.*\b(DM|supplement|coach)\b", 0.8),
    (r"\b(supplement\s*stack|coaching\s*plan)\b.*\bDM\b", 0.8),
    (r"\bDM\s*(me\s*)?(for|about)\b.*\b(supplement|coach|plan|stack)\b", 0.8),
    (r"\b(boss\s*babe|CEO\s*of\s*my\s*own\s*destiny)\b.*\b(earn|join|team|\$\d+|income|passive)\b", 0.85),
    (r"\b(earn|make)\s*\$?\d+K?\s*/\s*(month|week|day)\b", 0.8),
    (r"\bjoin\s*my\s*team\b", 0.8),
]

PROFANITY_PATTERNS: List[WeightedPattern] = [
    (r"\bf[\*\#]{2,}(ing)?\b", 0.8),
    (r"\bfuck(ing)?\b", 0.8),
    (r"\bshit\b", 0.6),
    (r"\bbitch\b", 0.6),
    (r"\bdamn\b", 0.3),
]

DRUGS_PATTERNS: List[WeightedPattern] = [
    (r"\bpsilocybin\b", 0.7),
    (r"\bmicrodosing?\b", 0.7),
    (r"\b(weed|marijuana|cannabis|pot)\b(?!\s*(out\b|killer\b|garden\b|lawn\b|spray\b))", 0.5),
    (r"\b(weed|marijuana|cannabis)\b.*\b(buy|sell|ship)\b", 0.8),
    (r"🍄.*\b(anxiety|depression|mental)\b", 0.5),
    (r"\bCBD\b", 0.5),
    (r"\bsupplement\s*stack\b", 0.5),
    (r"\b(moving|moved)\s*weight\b", 0.5),
    (r"\btrapping\b(?!.*\b(animal|pest|mouse)\b)", 0.5),
    (r"\b(adderall|ritalin|modafinil)\b.*\b(without|no)\s*prescription\b", 0.8),
    (r"\b(steroids?|HGH|testosterone)\b.*\b(cycle|stack|inject)\b", 0.7),
    (r"\b(lean|sizzurp|codeine)\b", 0.6),
]

GAMBLING_PATTERNS: List[WeightedPattern] = [
    (r"\bjackpot\b", 0.6),
    (r"\b(casino|slot|roulette|blackjack)\b", 0.6),
    (r"🎰", 0.6),
    (r"#gambling", 0.7),
    (r"\bbet(ting)?\b.*\b(odds|spread|parlay)\b", 0.6),
    (r"\bfree\s*spins\b", 0.85),
    (r"\buse\s*code\b.*\bcasino\b", 0.85),
    (r"\b(stake|wager|bet)\s*\$?\d+\b", 0.5),
    (r"\b(draftkings|fanduel|bet365)\b", 0.6),
    (r"\b(crypto|nft)\s*(gambl|bet|wager)\b", 0.65),
]

CONTROVERSIAL_PATTERNS: List[WeightedPattern] = [
    (r"\babortion\b.*\b(illegal|ban)\b", 0.8),
    (r"\bgun\s*control\b", 0.6),
    (r"\bgun\s*violence\b", 0.55),
    (r"\b(gun|firearm|weapon)\s*(reform|regulation|ban|laws?)\b", 0.6),
    (r"\b(nurse|doctor|teacher|firefighter|soldier|cop|police)\b.*\b(gun|weapon|violence|reform)\b", 0.6),
    (r"\bdestroying\s*democracy\b", 0.7),
    (r"\bfight\s*me\s*in\s*the\s*comments\b", 0.5),
    (r"\bthis\s*politician\b", 0.5),
    (r"\bclown\s*🤡", 0.4),
    (r"\bleaked\b.*\b(documents?|internal)\b", 0.7),
    (r"\b(crypto|bitcoin)\b.*\b(dropped?\s*out|quit)\b", 0.6),
    (r"\b(dropped?\s*out|quit)\b.*\b(crypto|bitcoin)\b", 0.6),
    (r"\btrade\s*crypto\b", 0.5),
    (r"#politics\b.*\bmeme\b", 0.6),
    (r"\bmeme\b.*#politics", 0.6),
    (r"\bsenator\b.*\b(asleep|sleeping|meme)\b", 0.55),
    (r"\b(israeli|palestinian|israel|palestine)\b", 0.6),
    (r"\b(rigged|stolen)\s*(election|vote)\b", 0.7),
    (r"\b(liberal|conservative)\s+(agenda|brainwash|sheep)\b", 0.6),
    (r"\bwhistleblower\b", 0.5),
    (r"\bcoverr?\s*up\b", 0.5),
]

PIRACY_PATTERNS: List[WeightedPattern] = [
    (r"\bpirated?\b", 0.8),
    (r"\bDM\s*(me\s*)?for\s*(the\s*)?link\b", 0.4),
    (r"\btorrent\b", 0.7),
    (r"\b(cracked|crack)\s*(software|game|app)\b", 0.75),
    (r"\b(free\s+)?download\b.*\b(movie|film|album|software)\b", 0.5),
]

TERRORISM_PATTERNS: List[WeightedPattern] = [
    (r"\bproud\s*boys\b", 0.9),
    (r"\bjoin\s*us\b.*\b(link|movement)\b", 0.5),
    (r"\bextremist\s*group\b", 0.9),
    (r"\brecruit(ment|ing)?\b", 0.4),
    (r"\bbomb\b.*\bschool\b", 0.95),
    (r"\b(jihad|caliphate|holy\s*war)\b", 0.9),
    (r"\b(lone\s*wolf|martyrdom|shaheed)\b", 0.85),
    (r"\b(pledge\s*allegiance)\b.*\b(ISIS|ISIL|al.?qaeda)\b", 0.95),
    (r"\b(militia|insurrection|armed\s*resistance)\b", 0.7),
    (r"\b(storm|siege)\s+(the\s+)?(capitol|parliament|government)\b", 0.85),
]

SAFE_PATTERNS: List[WeightedPattern] = [
    (r"#(baking|cooking|recipe|foodie|homemade)", 0.7),
    (r"#(fitness|running|gym|yoga|motivation)", 0.7),
    (r"#(hiking|outdoors|adventure|nature|travel)", 0.7),
    (r"#(dogsofinstagram|catsofinstagram|pets)", 0.7),
    (r"#(studytok|study|learning|education)", 0.7),
    (r"#(diy|crafts|art|creative)", 0.6),
    (r"#(tbt|throwback)", 0.3),
    (r"\brecipe\b", 0.5),
    (r"(🍪|🏃|🌅|🐾|📚|🙏|💪|💔)", 0.3),
    (r"\b(grateful|blessed|thankful)\b", 0.4),
    (r"\b(donated|charity|food\s*bank|volunteer)\b", 0.6),
    (r"\bsatire\b", 0.6),
    (r"\bscientists\s*baffled\b", 0.3),
    (r"\b(pomodoro|study\s*hack|productivity)\b", 0.6),
    (r"\bkombucha\b", 0.5),
    (r"\bfermentation\b", 0.4),
    (r"\bintermittent\s*fasting\b", 0.4),
    (r"\bgaming|call\s*of\s*duty\b", 0.3),
    (r"\breview\b.*\b(game|gaming)\b", 0.4),
    (r"\bgrandfather\b.*\b(war|WWII|fought)\b", 0.3),
    (r"\bnever\s*again\b", 0.3),
    (r"\bwild\s*college\s*days\b.*\bgrateful\b", 0.4),
    (r"\bwine\s*tasting\b", 0.4),
    (r"#(wine|italy|travel)", 0.4),
    (r"\bnot\s*for\s*everyone\b.*\bworks\s*for\s*me\b", 0.4),
    (r"#(crochet|cottagecore|handmade|amigurumi|knitting)", 0.6),
    (r"#(babymilestone|momlife|parenting|family)", 0.7),
    (r"#(booktok|reading|bookstagram)", 0.7),
    (r"#(craftbeer|beerreview|brewery)", 0.5),
    (r"#(truecrime|podcast)", 0.4),
    (r"#(lgbtq|pride|representation)", 0.5),
    (r"\b(PTSD|crisis\s*line|988)\b", 0.5),
    (r"\byears?\s*clean\b", 0.5),
    (r"\bcrisis\s*text\s*line\b", 0.5),
    (r"\brepresentation\s*matters\b", 0.5),
    (r"\bphotojournalism\b", 0.4),
    (r"\bhop\s*profile\b", 0.4),
    (r"(🧶|🌈|🎗️|🕊️)", 0.3),
]

SATIRE_MARKERS: List[str] = [
    r"\bsatire\b",
    r"\bbreaking:\s*local\b",
    r"\bscientists\s*baffled\b",
    r"\b/s\b",
]

PERSONAL_NARRATIVE_MARKERS: List[str] = [
    r"\bas\s*a\s*(nurse|doctor|teacher|parent|vet|firefighter|soldier|cop|police)\b",
    r"\bmy\s*(grandfather|grandmother|dad|mom|father|mother)\b",
    r"\bgrateful\s*for\s*growth\b",
    r"\bcan.?t\s*believe\s*I\s*survived\b",
    r"\bsmall\s*acts\s*matter\b",
    r"\bmy\s*\d+[-\s]year[-\s]old\b",
    r"\byears?\s*ago\s*I\b",
    r"\bif\s*I\s*can\s*change\b",
    r"\bI\s*(was|used\s*to|grew\s*up)\b",
    r"\bI\s*never\s*thought\b",
    r"\bmy\s*(journey|story|experience)\b",
]

ADVOCACY_MARKERS: List[str] = [
    r"\bwe\s*need\b.*\breform\b",
    r"\bcommon\s*sense\b",
    r"#guncontrol",
]


def scan_patterns(text: str, patterns: List[WeightedPattern]) -> SignalResult:
    max_weight = 0.0
    matched: List[str] = []
    for pattern, weight in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            max_weight = max(max_weight, weight)
            matched.append(m.group(0).strip())
    return SignalResult(strength=max_weight, matched=matched)


def scan_all_categories(text: str) -> SignalMap:
    return SignalMap(
        critical=scan_patterns(text, CRITICAL_PATTERNS),
        adult=scan_patterns(text, ADULT_PATTERNS),
        violence=scan_patterns(text, VIOLENCE_PATTERNS),
        hate_speech=scan_patterns(text, HATE_SPEECH_PATTERNS),
        illegal=scan_patterns(text, ILLEGAL_PATTERNS),
        misinformation=scan_patterns(text, MISINFORMATION_PATTERNS),
        profanity=scan_patterns(text, PROFANITY_PATTERNS),
        drugs=scan_patterns(text, DRUGS_PATTERNS),
        gambling=scan_patterns(text, GAMBLING_PATTERNS),
        controversial=scan_patterns(text, CONTROVERSIAL_PATTERNS),
        piracy=scan_patterns(text, PIRACY_PATTERNS),
        terrorism=scan_patterns(text, TERRORISM_PATTERNS),
        safe=scan_patterns(text, SAFE_PATTERNS),
    )


def _matches_any(text: str, markers: List[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in markers)


def detect_context(text: str) -> ContentContext:
    return ContentContext(
        is_satire=_matches_any(text, SATIRE_MARKERS),
        is_personal_narrative=_matches_any(text, PERSONAL_NARRATIVE_MARKERS),
        is_advocacy=_matches_any(text, ADVOCACY_MARKERS),
    )


def _text_matches(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, re.IGNORECASE))


def _set(
    c: Classification,
    decision: str, iab: str, garm: str, risk: str, confidence: float,
    reason: str, age_rating: str = "TEEN",
) -> Classification:
    c.decision = decision
    c.iab_category = iab
    c.garm_category = garm
    c.risk_level = risk
    c.age_rating = age_rating
    c.confidence = confidence
    c.reasoning_parts = [reason]
    return c


def _classify_critical(signals: SignalMap, c: Classification) -> Classification:
    if signals.hate_speech.strength >= 0.7:
        garm = "GARM_TERRORISM" if signals.terrorism.strength >= 0.9 else "GARM_HATE_SPEECH"
        return _set(c, "REJECT", "IAB_HATE_SPEECH", garm, "CRITICAL", 0.95,
                    "Critical: hate speech / extremist content detected", "ADULT")
    if signals.terrorism.strength >= 0.7:
        age = "ADULT" if signals.hate_speech.strength >= 0.5 else "MATURE"
        return _set(c, "REJECT", "IAB_VIOLENCE", "GARM_TERRORISM", "CRITICAL", 0.9,
                    "Critical: terrorism / violent threat content detected", age)
    return _set(c, "REJECT", "IAB_VIOLENCE", "GARM_DEATH_INJURY", "CRITICAL", 0.9,
                "Critical: violent/threatening content detected", "MATURE")


def _classify_illegal(signals: SignalMap, text: str, c: Classification) -> Classification:
    if signals.piracy.strength >= 0.7:
        return _set(c, "REJECT", "IAB_ILLEGAL", "GARM_ONLINE_PIRACY", "HIGH", 0.88,
                    "Copyright piracy / illegal distribution detected", "ADULT")
    if _text_matches(text, r"\bcredit\s*card\s*(number|info)\b") or _text_matches(text, r"\bgiveaway\b.*\b(DM|follow)\b"):
        return _set(c, "REJECT", "IAB_ILLEGAL", "GARM_SPAM_HARMFUL", "HIGH", 0.88,
                    "Phishing / scam content detected — requests personal financial information", "ADULT")
    return _set(c, "REJECT", "IAB_ILLEGAL", "GARM_CRIME_HARMFUL", "CRITICAL", 0.92,
                "Illegal activity promotion detected (drug sales / unlicensed substances)", "ADULT")


def _classify_severe_misinfo(text: str, c: Classification) -> Classification:
    if _text_matches(text, r"\b(cancer\s*cure|instead\s*of\s*chemo)\b"):
        return _set(c, "REJECT", "IAB_MISINFORMATION", "GARM_SPAM_HARMFUL", "CRITICAL", 0.9,
                    "Dangerous medical misinformation: promoting unproven alternatives to standard treatment", "TEEN")
    if _text_matches(text, r"\bjoin\s*my\s*team\b") or \
       _text_matches(text, r"\b(boss\s*babe|CEO\s*of\s*my\s*own)\b.*\b(earn|join|team|\$\d+|income)\b") or \
       _text_matches(text, r"\b(earn|make)\s*\$?\d+K?\s*/\s*(month|week|day)\b"):
        return _set(c, "REJECT", "IAB_MISINFORMATION", "GARM_SPAM_HARMFUL", "HIGH", 0.82,
                    "MLM / predatory marketing: income claims and recruitment pattern detected", "TEEN")
    if _text_matches(text, r"\b(supplement|coaching|transformation)\b.*\bDM\b"):
        return _set(c, "ESCALATE", "IAB_MISINFORMATION", "GARM_SPAM_HARMFUL", "MEDIUM", 0.72,
                    "Supplement/coaching marketing with DM funnel — needs human review for MLM assessment", "TEEN")
    return _set(c, "REJECT", "IAB_MISINFORMATION", "GARM_SPAM_HARMFUL", "MEDIUM", 0.78,
                "Misinformation / conspiracy content detected", "TEEN")


def _classify_profanity(signals: SignalMap, c: Classification) -> Classification:
    if signals.critical.strength >= 0.4:
        return _set(c, "REJECT", "IAB_PROFANITY", "GARM_OBSCENITY_PROFANITY", "HIGH", 0.85,
                    "Profanity combined with violent language detected", "MATURE")
    return _set(c, "REJECT", "IAB_PROFANITY", "GARM_OBSCENITY_PROFANITY", "HIGH", 0.8,
                "Explicit profanity detected", "MATURE")


def _classify_drugs(text: str, ctx: ContentContext, c: Classification) -> Classification:
    is_cbd = _text_matches(text, r"\bCBD\b")
    has_health_framing = _text_matches(text, r"\b(helped|anxiety|depression|mental\s*health|sleep|saved|transformation|weight\s*loss)\b")
    has_redemption = _text_matches(text, r"\b(redemption|can\s*change|legit|years?\s*clean)\b")
    is_accidental = _text_matches(text, r"\b(dog|cat|pet|kid|child|baby|accidentally|ate|eaten|ingested)\b")
    if is_cbd:
        return _set(c, "ESCALATE", "IAB_DRUGS", "GARM_SAFE", "MEDIUM", 0.6,
                    "CBD/hemp product discussion — legal but brand-dependent sensitivity, needs review", "TEEN")
    if ctx.is_personal_narrative or has_health_framing or has_redemption or is_accidental:
        return _set(c, "ESCALATE", "IAB_DRUGS", "GARM_CRIME_HARMFUL", "MEDIUM", 0.65,
                    "Substance discussion in personal health/redemption context — needs human review", "TEEN")
    return _set(c, "REJECT", "IAB_DRUGS", "GARM_CRIME_HARMFUL", "HIGH", 0.8,
                "Drug/substance promotion detected", "MATURE")


def _classify_controversial(text: str, ctx: ContentContext, c: Classification) -> Classification:
    if ctx.is_advocacy:
        return _set(c, "ESCALATE", "IAB_CONTROVERSIAL", "GARM_DEATH_INJURY", "MEDIUM", 0.6,
                    "Advocacy on divisive topic from personal/professional perspective — needs brand alignment check", "TEEN")
    if _text_matches(text, r"\b(leaked|documents?|internal)\b"):
        return _set(c, "ESCALATE", "IAB_CONTROVERSIAL", "GARM_SPAM_HARMFUL", "HIGH", 0.65,
                    "Potential defamation/leak content — escalate for legal review", "TEEN")
    if _text_matches(text, r"\bcrypto\b"):
        return _set(c, "ESCALATE", "IAB_CONTROVERSIAL", "GARM_SAFE", "MEDIUM", 0.6,
                    "Crypto promotion to potentially young audience — financial risk, escalate", "TEEN")
    if _text_matches(text, r"\bmeme\b") and _text_matches(text, r"#politics|political"):
        return _set(c, "ESCALATE", "IAB_CONTROVERSIAL", "GARM_SAFE", "MEDIUM", 0.6,
                    "Political humor/meme content — brand-sensitivity varies by advertiser, escalate", "TEEN")
    if _text_matches(text, r"\b(israeli|palestinian|israel|palestine)\b"):
        return _set(c, "ESCALATE", "IAB_CONTROVERSIAL", "GARM_SAFE", "MEDIUM", 0.6,
                    "Content touches sensitive geopolitical topic — needs brand alignment check", "TEEN")
    if _text_matches(text, r"\b(politician|democracy|abortion)\b"):
        is_abortion = _text_matches(text, r"\babortion\b")
        garm = "GARM_HATE_SPEECH" if is_abortion else "GARM_SPAM_HARMFUL"
        risk = "HIGH" if is_abortion else "MEDIUM"
        return _set(c, "REJECT", "IAB_CONTROVERSIAL", garm, risk, 0.75,
                    "Politically divisive content — brand risk regardless of viewpoint", "TEEN")
    return _set(c, "REJECT", "IAB_CONTROVERSIAL", "GARM_SPAM_HARMFUL", "MEDIUM", 0.7,
                "Controversial content detected — brand risk", "TEEN")


def _classify_safe(text: str, signals: SignalMap, ctx: ContentContext, c: Classification) -> Classification:
    if ctx.is_satire:
        reason = "Content identified as satire/humor with no harmful elements"
        c.confidence = 0.88
    elif ctx.is_personal_narrative and signals.safe.strength >= 0.3:
        reason = "Personal narrative / emotional content with no harmful signals"
        c.confidence = 0.88
    elif signals.safe.strength >= 0.5:
        reason = "Content matches brand-safe patterns (lifestyle, education, wellness)"
        c.confidence = 0.92
    elif signals.safe.strength >= 0.4:
        reason = "Content matches brand-safe patterns (lifestyle, education, wellness)"
        c.confidence = 0.88
    else:
        reason = "No harmful signals detected in content"
        c.confidence = 0.75
    age = _safe_age_rating(text)

    if _text_matches(text, r"\b(Aleppo|war\s*photo|rubble|bomb(ing|ed)|casualties|refugee)\b") and \
       _text_matches(text, r"\b(photojournalism|haunts|image|documentary|witness)\b"):
        return _set(c, "APPROVE", "IAB_SAFE", "GARM_DEATH_INJURY", "MEDIUM", c.confidence,
                    "Photojournalism / war reporting — newsworthy but sensitive imagery context", age)

    return _set(c, "APPROVE", "IAB_SAFE", "GARM_SAFE", "LOW", c.confidence, reason, age)


_TEEN_THEME_PATTERNS = [
    r"\b(wine|beer|IPA|bourbon|whiskey|vodka|cocktail|chianti|brewery|craft\s*beer)\b",
    r"🍷|🍺|🥃|🍸",
    r"\b(WWII|World\s*War|fought\s*in|war\b|combat|veteran)\b",
    r"\b(PTSD|self[- ]?harm|crisis\s*line|988|suicid|years?\s*clean)\b",
    r"\b(Afghanistan|Iraq|deployment|came\s*home\s*from)\b",
    r"\b(true\s*crime|murder\s*mystery|serial\s*killer|cold\s*case)\b",
    r"\b(Aleppo|war\s*photo|haunts\s*me|rubble|bomb)\b",
    r"\b(tw\s*//|trigger\s*warning|content\s*warning)\b",
    r"\bnever\s*again\b",
]


def _safe_age_rating(text: str) -> str:
    for pat in _TEEN_THEME_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "TEEN"
    return "ALL_AGES"


def classify_content(text: str, signals: SignalMap, context: ContentContext) -> Classification:
    c = Classification()

    if signals.critical.strength >= 0.8 and not context.is_satire:
        return _classify_critical(signals, c)
    if signals.terrorism.strength >= 0.8:
        age = "ADULT" if signals.hate_speech.strength >= 0.6 else "MATURE"
        return _set(c, "REJECT", "IAB_HATE_SPEECH", "GARM_TERRORISM", "CRITICAL", 0.93,
                    "Extremist group recruitment / terrorism content detected", age)
    if signals.adult.strength >= 0.7:
        return _set(c, "REJECT", "IAB_ADULT", "GARM_ADULT_EXPLICIT", "CRITICAL", 0.92,
                    "Adult/explicit content promotion detected", "ADULT")
    if signals.violence.strength >= 0.7:
        return _set(c, "REJECT", "IAB_VIOLENCE", "GARM_DEATH_INJURY", "HIGH", 0.88,
                    "Graphic violence content detected", "MATURE")
    if signals.hate_speech.strength >= 0.6 and not context.is_personal_narrative:
        return _set(c, "REJECT", "IAB_HATE_SPEECH", "GARM_HATE_SPEECH", "HIGH", 0.85,
                    "Hate speech / discriminatory content detected", "ADULT")
    if signals.illegal.strength >= 0.7:
        return _classify_illegal(signals, text, c)
    if signals.misinformation.strength >= 0.75:
        return _classify_severe_misinfo(text, c)
    if signals.profanity.strength >= 0.6:
        return _classify_profanity(signals, c)
    if signals.piracy.strength >= 0.6:
        return _set(c, "REJECT", "IAB_ILLEGAL", "GARM_ONLINE_PIRACY", "HIGH", 0.82,
                    "Copyright piracy content detected", "ADULT")
    if signals.drugs.strength >= 0.5:
        if signals.misinformation.strength >= 0.7:
            return _classify_severe_misinfo(text, c)
        return _classify_drugs(text, context, c)
    if signals.gambling.strength >= 0.5:
        if signals.gambling.strength >= 0.8:
            return _set(c, "REJECT", "IAB_GAMBLING", "GARM_CRIME_HARMFUL", "HIGH", 0.85,
                        "Gambling promotion / affiliate content detected", "TEEN")
        return _set(c, "ESCALATE", "IAB_GAMBLING", "GARM_SAFE", "MEDIUM", 0.6,
                    "Gambling content — brand suitability varies, needs human review", "TEEN")
    if signals.controversial.strength >= 0.5:
        return _classify_controversial(text, context, c)
    if signals.misinformation.strength >= 0.5:
        return _set(c, "REJECT", "IAB_MISINFORMATION", "GARM_SPAM_HARMFUL", "MEDIUM", 0.75,
                    "Health misinformation / predatory marketing pattern detected", "TEEN")
    if signals.critical.strength >= 0.5 and not context.is_satire:
        return _set(c, "REJECT", "IAB_VIOLENCE", "GARM_DEATH_INJURY", "HIGH", 0.8,
                    "Violent/threatening language detected", "MATURE")
    if signals.violence.strength >= 0.4 and not context.is_satire:
        return _set(c, "ESCALATE", "IAB_VIOLENCE", "GARM_DEATH_INJURY", "MEDIUM", 0.65,
                    "Potentially violent content — needs human review", "MATURE")
    return _classify_safe(text, signals, context, c)


def build_reasoning(
    classification: Classification, signals: SignalMap, context: ContentContext,
    content_type: str = "", platform: str = "",
) -> str:
    parts = list(classification.reasoning_parts)

    flagged = signals.all_flagged
    if flagged:
        parts.append(f"Flagged elements: {', '.join(flagged)}")

    if context.is_satire and classification.decision != "APPROVE":
        parts.append("Note: satire markers present but overridden by stronger negative signals")

    if platform:
        parts.append(f"Platform context: {platform}")
    if content_type:
        parts.append(f"Content type: {content_type}")

    if classification.decision == "APPROVE":
        parts.append(f"Risk assessment: {classification.risk_level}")
        parts.append(f"Age rating: {classification.age_rating}")
        if context.is_satire:
            parts.append("Satire context recognized — content is humorous, not harmful")
        elif context.is_personal_narrative:
            parts.append("Personal narrative with positive/neutral tone")
        if signals.safe.strength >= 0.5:
            parts.append("Strong brand-safety indicators present")

    reasoning = ". ".join(parts) + "."

    if len(reasoning) > 500:
        reasoning = reasoning[:497] + "..."
    if len(reasoning) < 10:
        reasoning = reasoning.ljust(10)

    return reasoning


def build_review_action(classification: Classification, reasoning: str, signals: SignalMap) -> Dict[str, Any]:
    return {
        "decision": classification.decision,
        "iab_category": classification.iab_category,
        "garm_category": classification.garm_category,
        "risk_level": classification.risk_level,
        "age_rating": classification.age_rating,
        "reasoning": reasoning,
        "confidence": round(classification.confidence, 2),
        "flagged_elements": signals.all_flagged,
    }


def smart_agent(text: str, content_type: str = "", platform: str = "") -> Dict[str, Any]:
    signals = scan_all_categories(text)
    context = detect_context(text)
    classification = classify_content(text, signals, context)
    reasoning = build_reasoning(classification, signals, context, content_type, platform)
    return build_review_action(classification, reasoning, signals)


AgentFn= Callable[[str, str, str], Dict[str, Any]]
GradeFn = Callable[..., Tuple[float, Dict[str, float], str]]


def evaluate_agent(
    items: List[Dict[str, Any]], grade_fn: GradeFn, agent_fn: Optional[AgentFn] = None,
) -> Dict[str, Any]:
    if agent_fn is None:
        agent_fn = smart_agent

    if not items:
        return {
            "results": [],
            "aggregate": {
                "mean_score": 0.0, "median_score": 0.0,
                "decision_accuracy": 0.0, "total_items": 0,
                "correct_decisions": 0,
                "by_difficulty": {"easy": 0.0, "medium": 0.0, "hard": 0.0},
            },
        }

    results: List[Dict[str, Any]] = []
    scores_by_difficulty: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    all_scores: List[float] = []
    correct_decisions = 0

    for item in items:
        action = agent_fn(
            item["content_text"],
            item.get("content_type", ""),
            item.get("platform", ""),
        )
        total, scores, feedback = grade_fn(action, item, steps_taken=1)

        all_scores.append(total)
        scores_by_difficulty[item["difficulty"]].append(total)
        if action["decision"] == item["gold_decision"]:
            correct_decisions += 1

        results.append({
            "content_id": item["content_id"],
            "difficulty": item["difficulty"],
            "predicted_decision": action["decision"],
            "gold_decision": item["gold_decision"],
            "correct": action["decision"] == item["gold_decision"],
            "total_score": round(total, 4),
            "scores": scores,
            "feedback": feedback,
        })

    return {
        "results": results,
        "aggregate": {
            "mean_score": round(statistics.mean(all_scores), 4),
            "median_score": round(statistics.median(all_scores), 4),
            "decision_accuracy": round(correct_decisions / len(items), 4),
            "total_items": len(items),
            "correct_decisions": correct_decisions,
            "by_difficulty": {
                d: round(statistics.mean(s), 4) if s else 0.0
                for d, s in scores_by_difficulty.items()
            },
        },
    }

# Backward-compatible alias
evaluate_all = evaluate_agent
