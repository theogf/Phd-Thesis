from manim import *

class AugmentationLaTeX(Scene):
    def construct(self):
        omega = MathTex("\omega").shift(UP)
        init_text = MathTex("p( {{ f }} {{ | }} {{ y }} )", "=", "{ {{ \prod_{i=1}^N }} \sigma(y_i f_i) {{ p(f) }} \over {{ p(y) }} }").set_color(BLACK)

        rewrite_sigma = MathTex("p( {{ f }} {{ | }} {{ y }} )", "=", "{ {{ \prod_{i=1}^N }} { {{ e^{y_if_i \over 2 } }} \over 2\cosh({|f_i| \over 2}) } {{ p(f) }} \over {{ p(y) }} }").set_color(BLACK)
        use_integral = MathTex("p( {{ f }} {{ | }} {{ y }} )", "=", "{ {{ \prod_{i=1}^N }} {{ e^{y_if_i \over 2 } }} \int_0^\infty {{ e^{-f_i^2\omega_i \over 2} }} {{ \mathrm{PG}(\omega_i|1, 0) }} d {{ \omega_i }} {{ p(f) }} \over {{ p(y) }} }").set_color(BLACK)
        final_text = MathTex("p( {{ f }}, \{ {{ \omega_i }} \}_{i=1}^N {{ | }} {{ y }} )", "=",  "{ {{ \prod_{i=1}^N }} {{ e^{y_if_i \over 2} }} {{ e^{-f_i^2\omega_i \over 2} }} {{ \mathrm{PG}(\omega_i|1, 0) }} {{ p(f) }} \over {{ p(y) }}}").set_color(BLACK)
        cond_f = MathTex("p(f|y, \{\omega_i\}_{i=1}^N) \propto {{ \prod_{i=1}^N }} {{ e^{y_if_i \over 2} }} {{ e^{-f_i^2\omega_i \over 2} }} {{ p(f) }}").set_color(BLACK)
        cond_f.next_to(final_text, DOWN)

        scale = Text("Representation as a scale-mixture").set_color(BLACK)
        aug = Text("Latent variable augmentation").set_color(BLACK)
        scale.next_to(rewrite_sigma, UP)        
        aug.next_to(final_text, UP)        
        self.add(init_text)
        self.add(index_labels(init_text[0]))
        # self.play(Write(init_text))
        self.wait()
        self.play(TransformMatchingTex(init_text, rewrite_sigma, fade_transform_mismatches=True), run_time=3)
        # self.add(rewrite_sigma)
        self.wait()        
        self.play(TransformMatchingTex(rewrite_sigma, use_integral, transform_mismatches=True), FadeIn(scale), run_time=3)
        self.wait()
        self.play(TransformMatchingTex(use_integral, final_text, transform_mismatches=False), FadeOut(scale), FadeIn(aug), run_time=3)
        self.wait()
        self.play(TransformMatchingTex(final_text.copy(), cond_f, transform_mismatches=False), FadeOut(aug), run_time=3)
