document.addEventListener("DOMContentLoaded", (event) => {
    // Register GSAP plugins
    gsap.registerPlugin(ScrollTrigger, MotionPathPlugin);

    // Initial setup to make sure plane is visible before animation starts
    gsap.set(".airplane", {
        xPercent: -50,
        yPercent: -50,
        transformOrigin: "center center"
    });

    // Create the path animation linked to the scroll position
    gsap.to(".airplane", {
        scrollTrigger: {
            trigger: "body",
            start: "top top",      // Start when body top hits viewport top
            end: "bottom bottom",  // End when body bottom hits viewport bottom
            scrub: 1.5,            // Smooth scrubbing (1.5 seconds smoothing)
        },
        motionPath: {
            path: "#flight-path",  // The ID of the SVG path
            align: "#flight-path", // Align the element to the path
            autoRotate: 90,        // Automatically rotate the element to match the path's angle (adjusted by 90deg to orient the SVG correctly)
            alignOrigin: [0.5, 0.5] // Align center of the airplane to the path
        },
        ease: "power1.inOut"
    });

    // Animate text sections as they enter the viewport
    const cards = gsap.utils.toArray('.glass-card');
    
    cards.forEach((card, i) => {
        gsap.from(card, {
            scrollTrigger: {
                trigger: card,
                start: "top 80%", // trigger when the top of the card is 80% down the screen
                toggleActions: "play none none reverse" 
            },
            y: 50,
            opacity: 0,
            duration: 1,
            ease: "back.out(1.7)"
        });
    });

});
