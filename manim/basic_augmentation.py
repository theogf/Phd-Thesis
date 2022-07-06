from manim import *

class AugmentationLaTeX(Scene):
    def construct(self):
        omega = MathTex("\omega").shift(UP)
        init_text = MathTex(r"p({{\theta}}|{{x}})", "=", r"{p(x|\theta}}{{)}}{{p(\theta)}} \over {{p(x)}}}").set_color(BLACK)
        # init_text = MathTex("p({{f}}|{{y}})", "=", r"\frac{p(y|f)p(f)}{p(y)}")
        int_text = MathTex(r"p({{\theta}}|{{x}})", "=", r"{\int_0^\infty p({{x|\theta}},{{\omega}}{{)}} {{p(\omega)}} d{{\omega}}{{p(\theta)}} \over {{p(x)}}}").set_color(BLACK)
        final_text = MathTex(r"p({{\theta}},{{\omega}}|{{x}})", "=",  r"{p({{x|\theta}}, {{\omega}}{{)}}{{p(\omega)}}{{p(\theta)}} \over {{p(x)}}}").set_color(BLACK)
        # self.add(init_text)
        self.play(Write(init_text))
        self.play(TransformMatchingTex(Group(init_text, omega), int_text), run_time=3)
        self.wait(3)
        self.play(TransformMatchingTex(int_text, final_text), run_time=3)
        self.wait(3)