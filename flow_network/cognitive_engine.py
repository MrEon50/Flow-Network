import torch
import collections
import time
from typing import Dict, List, Set, Tuple, Optional

class KnowledgeGraph:
    """
    Pamięć Semantyczna (Graf Wiedzy).
    Przechowuje utrwalone abstrakcyjne informacje o świecie oraz relacje pomiędzy nimi.
    """
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)

    def add_fact(self, subject: str, predicate: str, object_node: str):
        self.nodes.add(subject)
        self.nodes.add(object_node)
        self.edges[subject].append((predicate, object_node))

    def retrieve_context(self, query_entities: List[str]) -> str:
        """Pobiera skorelowane fakty dla danego zbioru węzłów i układa w tekst kontekstowy"""
        context = []
        for entity in query_entities:
            if entity in self.edges:
                for predicate, obj in self.edges[entity]:
                    context.append(f"{entity} {predicate} {obj}.")
        return " ".join(context) if context else ""

class EpisodicBuffer:
    """
    Pamięć Epizodyczna z Ograniczoną Pojemnością.
    Śledzi co się wydarzyło przed chwilą (Ostatnie konwersacje / logi).
    """
    def __init__(self, capacity: int = 5):
        self.events = collections.deque(maxlen=capacity)

    def add_event(self, event: str):
        self.events.append(event)

    def get_recent_history(self) -> str:
        return " | ".join(self.events)

class CognitiveFlowAgent:
    """
    Agent Kognitywny spinający architekturę FlowNetwork (Pamięć Robocza) 
    z pamięcią Długoterminową (Semantyczny Graf) oraz Krótkoterminową (Epizodyczną).
    
    Zamiast zmuszać sieć do "zapamiętywania", agent wstrzykuje odpowiednie 
    fakty z zewnętrznego twardego dysku do liniowego okna przepływu (O(N)).
    """
    def __init__(self, flow_model, stoi: Dict[str, int], itos: Dict[int, str], device='cpu'):
        self.brain = flow_model  # FlowNetwork (Głównik logiki / Reasoning Engine)
        self.stoi = stoi
        self.itos = itos
        self.device = device
        
        self.semantic_memory = KnowledgeGraph()
        self.episodic_memory = EpisodicBuffer(capacity=5)

    def _encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, self.stoi.get(' ', 0)) for c in text]

    def _decode(self, tokens: List[int]) -> str:
        return ''.join([self.itos.get(i, '?') for i in tokens])

    def perceive_and_think(self, user_input: str, extracted_keywords: List[str] = None) -> str:
        """
        Pętla Kognitywna: 
        1. Rozpoznanie zapytania.
        2. Sięgnięcie do bazy wiedzy w celu segregacji faktów.
        3. Scalenie Pamięci Epizodycznej, Semantycznej z Pamięcią Roboczą strumienia wejściowego.
        4. Wypchnięcie predykcji.
        """
        # 1. Zapisujemy fakt, że użytkownik nas o coś spytał
        self.episodic_memory.add_event(f"User asked: {user_input[:20]}...")
        
        # 2. Jeśli agent rozpoznaje jakieś klucze, szuka twardych faktów na swoim "dysku twardym" (RAG)
        if extracted_keywords is None:
            extracted_keywords = []
            # Prosta symulacja ekstrakcji np. wszystkich wyrażeń z dużej litery
            for word in user_input.split():
                if word and word[0].isupper():
                    extracted_keywords.append(word)
                    
        hard_facts = self.semantic_memory.retrieve_context(extracted_keywords)
        
        # 3. ZŁOTA INŻYNIERIA (Dynamiczny Kontekst)
        # Zamiast modelu, który czyta 10 tysięcy tokenów dla jednego szczegółu (O(N^2) zniszczenie),
        # podajemy tylko wyselekcjonowaną pamięć prosto w liniowy ruter sieci:
        
        system_prompt = ""
        if hard_facts:
            system_prompt += f"[SEMANTIC_RAG: {hard_facts}] "
            
        recent_history = self.episodic_memory.get_recent_history()
        system_prompt += f"[EPISODIC: {recent_history}] "
        
        # Ostateczny strumień połączony z zewnętrzną pętlą nauki:
        full_working_memory = system_prompt + user_input
        
        # 4. Przekazanie do Sieci Flow w celu logicznych wniosków i predykcji autouzupełnienia
        return self._generate_from_brain(full_working_memory, max_tokens=100)

    @torch.no_grad()
    def _generate_from_brain(self, full_context: str, max_tokens: int) -> str:
        self.brain.eval()
        encoded = self._encode(full_context)
        idx = torch.tensor(encoded, dtype=torch.long, device=self.device).unsqueeze(0)

        for _ in range(max_tokens):
            # Model Flow pobiera kontekst. Ze względu na wrodzoną strukturę O(N), 
            # może błyskawicznie "przelecieć" przez zaserwowany z zewnątrz RAG bez nagłego przegrzania GPU
            idx_cond = idx[:, -256:] 
            logits, _ = self.brain(idx_cond)
            logits = logits[:, -1, :] 
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        
        # Wydobywamy z indeksu nową predykcję ukrywając zaaplikowany z zewnątrz RAG
        out = self._decode(idx[0].tolist())
        self.brain.train()
        
        # Zapamiętujemy że "Pomyśleliśmy to" do bufora
        generated_part = out[len(full_context):]
        self.episodic_memory.add_event(f"I generated text length: {len(generated_part)}")
        
        return generated_part

    def dream(self):
        """
        Pętla Snu (Consolidation Layer).
        Pozwala agentowi konsolidować pamięć epizodyczną do grafu wiedzy
        w wolnym czasie bez udziału użytkownika. (Podstawa pętli nauki Agentów LLM).
        """
        history = self.episodic_memory.get_recent_history()
        return f"W czasie snu agent analizuje ostatnio przetrwonione epizody: '{history}' w celu zapisania nowych twardych definicji."
