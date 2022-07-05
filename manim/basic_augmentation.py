from manim import *

class AugmentationLaTeX(Scene):
    def construct(self):
        omega = MathTex("\omega").shift(UP)
        init_text = MathTex("p({{f}}|{{y}})", "=", "{p({{y|f}}){{p(f)}} \over {{p(y)}}}").set_color(BLACK)
        # init_text = MathTex("p({{f}}|{{y}})", "=", r"\frac{p(y|f)p(f)}{p(y)}")
        final_text = MathTex(r"p({{f}},{{\omega}}|{{y}})", "=",  "{p({{y|f}}, {{\omega}})p({{\omega}}){{p(f)}} \over {{p(y)}}}").set_color(BLACK)
        # self.add(init_text)
        self.play(Write(init_text))
        self.wait()
        self.play(TransformMatchingTex(Group(init_text, omega), final_text), run_time=3)