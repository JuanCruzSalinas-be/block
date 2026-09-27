const header = document.getElementById("header");
const panels = document.querySelectorAll(".mega");
const panelButtons = document.querySelectorAll("[data-panel]");
const search = document.getElementById("search");
const searchToggle = document.getElementById("searchToggle");
const mobileMenu = document.getElementById("mobileMenu");
const menuToggle = document.getElementById("menuToggle");


// Header background on scroll
const onScroll = () => {
  const scrolled = window.scrollY > 40;

  header.classList.toggle("is-scrolled", scrolled);
};


window.addEventListener(
  "scroll",
  onScroll,
  { passive: true }
);

onScroll();


// Sync header open state
const syncOpenState = () => {

  const anyOpen =
    [...panels].some((panel) => !panel.hidden) ||
    !search.hidden ||
    !mobileMenu.hidden;

  header.classList.toggle(
    "is-open",
    anyOpen
  );

  onScroll();
};


// Close all navigation panels
const closeAll = () => {

  panels.forEach((panel) => {
    panel.hidden = true;
  });


  panelButtons.forEach((button) => {
    button.setAttribute(
      "aria-expanded",
      "false"
    );
  });


  search.hidden = true;

  mobileMenu.hidden = true;


  menuToggle.setAttribute(
    "aria-expanded",
    "false"
  );


  document.body.classList.remove(
    "no-scroll"
  );


  syncOpenState();
};


// Mega menu panels
panelButtons.forEach((button) => {

  button.addEventListener(
    "click",
    () => {

      const panel = document.getElementById(
        `panel-${button.dataset.panel}`
      );


      const wasOpen = !panel.hidden;


      closeAll();


      if (!wasOpen) {

        panel.hidden = false;

        button.setAttribute(
          "aria-expanded",
          "true"
        );

      }


      syncOpenState();

    }
  );

});


// Search
searchToggle.addEventListener(
  "click",
  () => {

    const wasOpen = !search.hidden;


    closeAll();


    if (!wasOpen) {

      search.hidden = false;

      search
        .querySelector("input")
        .focus();

    }


    syncOpenState();

  }
);


// Mobile menu
menuToggle.addEventListener(
  "click",
  () => {

    const wasOpen = !mobileMenu.hidden;


    closeAll();


    if (!wasOpen) {

      mobileMenu.hidden = false;


      menuToggle.setAttribute(
        "aria-expanded",
        "true"
      );


      document.body.classList.add(
        "no-scroll"
      );

    }


    syncOpenState();

  }
);


// Escape closes open menus
document.addEventListener(
  "keydown",
  (event) => {

    if (event.key === "Escape") {
      closeAll();
    }

  }
);


// Clicking outside closes menus
document.addEventListener(
  "click",
  (event) => {

    if (
      !header.contains(event.target) &&
      !mobileMenu.contains(event.target)
    ) {

      closeAll();

    }

    else if (
      event.target.closest(
        ".mega a, .mobile-menu a"
      )
    ) {

      closeAll();

    }

  }
);


// Reveal on scroll
const revealObserver =
  new IntersectionObserver(

    (entries) => {

      entries.forEach((entry) => {

        if (entry.isIntersecting) {

          entry.target.classList.add(
            "is-visible"
          );


          revealObserver.unobserve(
            entry.target
          );

        }

      });

    },

    {
      threshold: 0.15
    }

  );


document
  .querySelectorAll(".reveal")
  .forEach((element) => {

    revealObserver.observe(element);

  });


// Count-up stats
const countObserver =
  new IntersectionObserver(

    (entries) => {

      entries.forEach((entry) => {

        if (!entry.isIntersecting) {
          return;
        }


        const element =
          entry.target;


        const target =
          Number(element.dataset.count);


        const suffix =
          element.dataset.suffix || "";


        const start =
          performance.now();


        const duration =
          1600;


        const tick = (now) => {

          const progress =
            Math.min(
              (now - start) / duration,
              1
            );


          const eased =
            1 -
            Math.pow(
              1 - progress,
              3
            );


          element.textContent =
            Math.round(
              target * eased
            ).toLocaleString() +
            suffix;


          if (progress < 1) {

            requestAnimationFrame(
              tick
            );

          }

        };


        requestAnimationFrame(
          tick
        );


        countObserver.unobserve(
          element
        );

      });

    },

    {
      threshold: 0.5
    }

  );


document
  .querySelectorAll("[data-count]")
  .forEach((element) => {

    countObserver.observe(
      element
    );

  });


// Current year
document
  .getElementById("year")
  .textContent =
    new Date().getFullYear();
