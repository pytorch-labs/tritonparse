import React from 'react';

interface ExternalLinkProps {
  href: string;
  title: string;
  text: string;
  icon: React.ReactNode;
}

/**
 * A reusable external link component with consistent styling.
 * Used for navigation links in the header (GitHub, Wiki, Internal Wiki, etc.)
 */
const ExternalLink: React.FC<ExternalLinkProps> = ({ href, title, text, icon }) => {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-gray-600 hover:text-gray-800 transition-colors flex items-center space-x-1"
      title={title}
    >
      {icon}
      <span className="text-sm font-medium">{text}</span>
    </a>
  );
};

export default ExternalLink;
