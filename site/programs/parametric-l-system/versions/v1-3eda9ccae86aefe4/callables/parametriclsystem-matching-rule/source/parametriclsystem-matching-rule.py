def _matching_rule(
        self, symbol: str, left: str | None, right: str | None
    ) -> RuleResult | RuleCallable | None:
        best: RuleResult | RuleCallable | None = None
        best_specificity = -1
        for key, rule in self.rules.items():
            if isinstance(key, str):
                if key == symbol and best_specificity < 0:
                    best = rule
                    best_specificity = 0
                continue
            required_left, required_symbol, required_right = key
            if required_symbol != symbol:
                continue
            if required_left is not None and required_left != left:
                continue
            if required_right is not None and required_right != right:
                continue
            specificity = int(required_left is not None) + int(required_right is not None)
            if specificity > best_specificity:
                best = rule
                best_specificity = specificity
        return best